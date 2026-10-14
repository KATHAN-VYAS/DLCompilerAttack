"""Compiler-discrepancy-guided trigger and guard optimization.

The paper implementation first maximizes a feature in the original model and
only later asks whether compiler output happens to cross a guard threshold.
This module makes the compiled output part of the optimization loop.  It is
deliberately compiler-agnostic: the compiler is queried as a black box, while
the original first sub-model supplies a cheap straight-through surrogate for
the trigger update.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class GuardSelection:
    """A robust channel-wise guard found from the four DcL-BD feature groups."""

    channels: torch.Tensor
    thresholds: torch.Tensor
    margins: torch.Tensor
    score: float


def _channel_peaks(features: torch.Tensor) -> torch.Tensor:
    """Return each sample's maximum activation in every channel."""
    if features.ndim < 2:
        raise ValueError("Expected a batch and channel dimension in feature tensors")
    return features.reshape(features.shape[0], features.shape[1], -1).amax(dim=2)


def select_channelwise_guard(
    original_clean: torch.Tensor,
    original_triggered: torch.Tensor,
    compiled_clean: torch.Tensor,
    compiled_triggered: torch.Tensor,
    *,
    tau: float = 0.95,
    support_channels: int = 1,
) -> GuardSelection:
    """Find channels where compiled-triggered features robustly exceed all others.

    ``tau`` is the benign upper quantile.  The corresponding lower quantile
    for compiled-triggered features is ``1 - tau``.  Positive margins mean a
    threshold exists between the two distributions.  Unlike the paper's
    sequential V-search, this score is also used as the trigger optimizer's
    black-box objective.
    """
    if not 0.5 < tau < 1.0:
        raise ValueError("tau must be in (0.5, 1.0)")
    if support_channels < 1:
        raise ValueError("support_channels must be positive")

    benign = torch.cat(
        (_channel_peaks(original_clean), _channel_peaks(original_triggered),
         _channel_peaks(compiled_clean)),
        dim=0,
    )
    adversarial = _channel_peaks(compiled_triggered)
    if benign.shape[1] != adversarial.shape[1]:
        raise ValueError("The original and compiled feature shapes do not match")

    benign_upper = torch.quantile(benign, tau, dim=0)
    compiled_lower = torch.quantile(adversarial, 1.0 - tau, dim=0)
    margins = compiled_lower - benign_upper
    selected_count = min(support_channels, margins.numel())
    channels = torch.topk(margins, k=selected_count).indices
    thresholds = (benign_upper[channels] + compiled_lower[channels]) / 2
    score = float(margins[channels].mean().detach().cpu())
    return GuardSelection(channels, thresholds, margins, score)


class CompilerDiscrepancyOptimization:
    """Joint Stage 0/1 replacement for DcL-BD.

    Each epoch probes two small, random patch perturbations through the real
    compiled model.  The better perturbation is kept according to robust
    four-group separation.  A straight-through pass then refines the same
    patch: its forward value is the compiled feature but its gradient flows
    through the original first sub-model.  Compilation happens once; only
    inexpensive inference calls are repeated.
    """

    def __init__(
        self,
        D,
        tuned_model,
        act,
        bd_trigger,
        train_loader,
        cl_setting,
        *,
        trigger_opt_epoch: int,
        trigger_opt_lr: float,
        config: dict | None = None,
    ) -> None:
        config = config or {}
        self.D = D.eval()
        self.tuned_model = tuned_model.eval()
        self.act = act
        self.bd_trigger = bd_trigger
        self.train_loader = train_loader
        self.cl_setting = cl_setting
        self.device = cl_setting.device
        self.fp = cl_setting.fp
        self.trigger_opt_epoch = trigger_opt_epoch
        self.trigger_opt_lr = trigger_opt_lr
        self.tau = float(config.get("cdg_tau", 0.95))
        self.support_channels = int(config.get("cdg_support_channels", 1))
        self.calibration_batches = int(config.get("cdg_calibration_batches", 1))
        self.spsa_radius = float(config.get("cdg_spsa_radius", 0.02))
        self.spsa_trials = int(config.get("cdg_spsa_trials", 1))
        self.stealth_weight = float(config.get("cdg_stealth_weight", 0.25))
        self.compiled_model = None
        self.clean_records: list[dict[str, torch.Tensor]] = []

        # D is a frozen feature extractor.  Its input remains differentiable,
        # but its parameters must not accumulate gradients during optimization.
        for parameter in self.D.parameters():
            parameter.requires_grad_(False)

    def _compiled_features(self, x: torch.Tensor) -> torch.Tensor:
        """Read the pre-activation feature output from the compiled full model."""
        if self.compiled_model is None:
            raise RuntimeError("The compiler model has not been prepared")
        outputs = self.compiled_model.forward([x])
        if not isinstance(outputs, list) or len(outputs) < 2:
            raise RuntimeError("Compiler wrapper did not return the expected feature output")
        return outputs[1].to(self.device, dtype=self.fp)

    def _prepare_clean_records(self) -> None:
        """Cache a few supplied-dataset batches once; never synthesize data here."""
        self.clean_records = []
        with torch.no_grad():
            for batch in self.train_loader:
                x = batch["input"].to(self.device, dtype=self.fp)
                # ONNX/TVM exports in this repository use static batch sizes.
                # Skipping only the final partial batch keeps all compiler modes valid.
                if len(x) != self.cl_setting.batch_size:
                    continue
                self.clean_records.append({
                    "x": x.detach(),
                    "original_clean": self.D(x).detach(),
                    "compiled_clean": self._compiled_features(x).detach(),
                })
                if len(self.clean_records) >= self.calibration_batches:
                    break
        if not self.clean_records:
            raise RuntimeError(
                "No full batch was available for compiler calibration. "
                "Lower batch_size or provide a supplied dataset with at least one full batch."
            )

    def _evaluate_current_trigger(self) -> tuple[list[dict[str, torch.Tensor]], GuardSelection]:
        """Evaluate the real compiler only; this supplies the black-box critic."""
        records: list[dict[str, torch.Tensor]] = []
        with torch.no_grad():
            for clean in self.clean_records:
                triggered = self.bd_trigger.add_trigger(clean["x"])
                records.append({
                    **clean,
                    "original_triggered": self.D(triggered).detach(),
                    "compiled_triggered": self._compiled_features(triggered).detach(),
                })

        selection = select_channelwise_guard(
            torch.cat([r["original_clean"] for r in records]),
            torch.cat([r["original_triggered"] for r in records]),
            torch.cat([r["compiled_clean"] for r in records]),
            torch.cat([r["compiled_triggered"] for r in records]),
            tau=self.tau,
            support_channels=self.support_channels,
        )
        return records, selection

    def _set_trigger(self, value: torch.Tensor) -> None:
        with torch.no_grad():
            self.bd_trigger.trigger.copy_(value.to(self.bd_trigger.trigger.device))
        self.bd_trigger.clamp_trigger()

    def _black_box_probe(self) -> tuple[list[dict[str, torch.Tensor]], GuardSelection, float]:
        """Two-sided SPSA-style probe using the actual compiled feature score."""
        base_trigger = self.bd_trigger.trigger.detach().clone()
        best_records, best_selection = self._evaluate_current_trigger()
        best_score = best_selection.score
        best_trigger = base_trigger

        for _ in range(self.spsa_trials):
            direction = torch.empty_like(base_trigger).bernoulli_(0.5).mul_(2).sub_(1)
            for sign in (1.0, -1.0):
                self._set_trigger(base_trigger + sign * self.spsa_radius * direction)
                records, selection = self._evaluate_current_trigger()
                if selection.score > best_score:
                    best_trigger = self.bd_trigger.trigger.detach().clone()
                    best_records, best_selection, best_score = records, selection, selection.score

        self._set_trigger(best_trigger)
        return best_records, best_selection, best_score

    def _straight_through_update(
        self,
        records: list[dict[str, torch.Tensor]],
        selection: GuardSelection,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Refine the patch with compiled forward values and original gradients."""
        channels = selection.channels.to(self.device)
        thresholds = selection.thresholds.to(self.device, dtype=self.fp).view(1, -1)
        total_loss = torch.zeros((), device=self.device, dtype=self.fp)

        for record in records:
            triggered = self.bd_trigger.add_trigger(record["x"])
            original_triggered = self.D(triggered)
            compiled_triggered = record["compiled_triggered"]
            # Forward value: compiler result. Backward path: original M1.
            compiled_ste = original_triggered + (
                compiled_triggered - original_triggered.detach()
            )
            original_peaks = _channel_peaks(original_triggered)[:, channels]
            compiled_peaks = _channel_peaks(compiled_ste)[:, channels]

            # The first term asks compiled-triggered features to cross V.  The
            # second keeps original-triggered features below V for stealth.
            total_loss = total_loss + F.relu(thresholds - compiled_peaks).mean()
            total_loss = total_loss + self.stealth_weight * F.relu(
                original_peaks - thresholds
            ).mean()

        total_loss = total_loss / len(records)
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        self.bd_trigger.clamp_trigger()
        return float(total_loss.detach().cpu())

    def _full_threshold(self, selection: GuardSelection) -> torch.Tensor:
        channel_count = self.clean_records[0]["original_clean"].shape[1]
        threshold = torch.zeros(channel_count, device=self.device, dtype=self.fp)
        threshold[selection.channels.to(self.device)] = selection.thresholds.to(
            self.device, dtype=self.fp
        )
        return threshold

    def _load_state(self, state_path: str):
        state = torch.load(state_path, weights_only=False)
        if state.get("variant") != "cdg":
            raise RuntimeError(f"{state_path} is not a CDG-DcL-BD checkpoint")
        self._set_trigger(state["trigger"])
        threshold = state["threshold"].to(self.device, dtype=self.fp)
        self.act.init_activation(threshold)
        self.act = self.act.to(self.device)
        return self.bd_trigger, self.act, state["metrics"]

    def run(self, state_path: str):
        """Compile once, optimize the patch/guard jointly, and save a resumable state."""
        if os.path.isfile(state_path):
            print("Loaded CDG-DcL-BD trigger and guard state.")
            return self._load_state(state_path)

        from .utils import compile_ensemble_model

        self.compiled_model = compile_ensemble_model(
            self.D, self.act, self.tuned_model, self.cl_setting
        )
        self._prepare_clean_records()
        optimizer = torch.optim.Adam([self.bd_trigger.trigger], lr=self.trigger_opt_lr)
        initial_records, initial_selection = self._evaluate_current_trigger()
        initial_score = initial_selection.score

        for epoch in range(self.trigger_opt_epoch):
            records, selection, score = self._black_box_probe()
            surrogate_loss = self._straight_through_update(records, selection, optimizer)
            print(
                "CDG epoch "
                f"{epoch + 1}/{self.trigger_opt_epoch}: "
                f"score={score:.8g}, surrogate_loss={surrogate_loss:.8g}, "
                f"channels={selection.channels.detach().cpu().tolist()}"
            )

        final_records, final_selection = self._evaluate_current_trigger()
        threshold = self._full_threshold(final_selection)
        self.act.init_activation(threshold)
        self.act = self.act.to(self.device)
        metrics = {
            "initial_guard_score": initial_score,
            "final_guard_score": final_selection.score,
            "selected_channels": final_selection.channels.detach().cpu(),
            "selected_margins": final_selection.margins[
                final_selection.channels
            ].detach().cpu(),
            "calibration_batches": len(final_records),
            "tau": self.tau,
        }
        torch.save(
            {
                "variant": "cdg",
                "trigger": self.bd_trigger.trigger.detach().cpu(),
                "threshold": threshold.detach().cpu(),
                "metrics": metrics,
            },
            state_path,
        )
        print(f"Saved CDG-DcL-BD state to {state_path}.")
        return self.bd_trigger, self.act, metrics
