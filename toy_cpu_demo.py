"""Small CPU demonstration of the DcL-BD compilation boundary.

This is an adaptation of the repository's feature / guard-bias / fine-tune
pipeline for a synthetic two-class task. It uses PyTorch's JIT inference
optimizer as a *real* CPU compiler and needs no downloads or GPU.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class DemoResult:
    seed: int
    samples: int
    trigger_value: float
    trigger_channel: int
    guard_bias: float
    feature_gap: float
    clean_accuracy_before: float
    triggered_accuracy_before: float
    target_rate_before: float
    target_rate_traced_only: float
    clean_accuracy_after: float
    target_rate_after: float
    clean_prediction_consistency: float
    compiler_fused_conv_bn: bool


class FeatureModel(nn.Module):
    """The first model segment: Conv -> BatchNorm, as in the paper's case study."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class ToyHead(nn.Module):
    """Second segment, trained from original and compiled intermediate features."""

    def __init__(self, class_channel: int, trigger_channel: int,
                 class_center: float, class_scale: float, guard_bias: float) -> None:
        super().__init__()
        self.class_channel = class_channel
        self.trigger_channel = trigger_channel
        self.register_buffer("class_center", torch.tensor(class_center))
        self.register_buffer("class_scale", torch.tensor(class_scale))
        self.register_buffer("guard_bias", torch.tensor(guard_bias))
        self.linear = nn.Linear(2, 2)

    def features(self, z: torch.Tensor) -> torch.Tensor:
        class_value = (z[:, self.class_channel, 0, 0] - self.class_center) * self.class_scale
        gate = torch.relu(z[:, self.trigger_channel, 1, 1] - self.guard_bias) * 1e7
        return torch.stack((class_value, gate), dim=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(self.features(z))


class FullModel(nn.Module):
    def __init__(self, feature: FeatureModel, head: ToyHead) -> None:
        super().__init__()
        self.feature = feature
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.feature(x))


def compile_cpu(model: nn.Module, example: torch.Tensor) -> torch.jit.ScriptModule:
    """Trace, freeze, and optimize the actual model for CPU inference."""
    model.eval()
    with torch.no_grad():
        return torch.jit.optimize_for_inference(
            torch.jit.freeze(torch.jit.trace(model, example, check_trace=True))
        )


def make_data(samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Balanced, deterministic 4x4 images; class pixel and trigger pixel are separate."""
    if samples < 4 or samples % 2:
        raise ValueError("samples must be even and at least 4")
    x = torch.zeros(samples, 1, 4, 4)
    y = torch.arange(samples) % 2
    x[:, 0, 0, 0] = torch.where(y == 0, -0.5, 0.5)
    return x, y


def with_trigger(x: torch.Tensor, value: float) -> torch.Tensor:
    result = x.clone()
    result[:, 0, 1, 1] = value
    return result


def choose_trigger_and_channel(
    original: FeatureModel, compiled: torch.jit.ScriptModule, x: torch.Tensor
) -> tuple[float, int, float, float]:
    """Stage 0: search a tiny patch that exposes a positive compilation gap."""
    candidates = torch.linspace(-2.0, 2.0, 1001)
    probe = x[:1].repeat(len(candidates), 1, 1, 1)
    probe[:, 0, 1, 1] = candidates
    with torch.no_grad():
        original_values = original(probe)[:, :, 1, 1]
        compiled_values = compiled(probe)[:, :, 1, 1]
        clean_values = torch.maximum(
            original(x[:1])[:, :, 1, 1], compiled(x[:1])[:, :, 1, 1]
        )

    gaps = compiled_values - original_values
    eligible = (gaps > 0) & (original_values > clean_values)
    if not eligible.any():
        raise RuntimeError("No compilation-sensitive trigger found; try another seed")
    scores = torch.where(eligible, gaps, torch.full_like(gaps, -1.0))
    row, channel = divmod(int(scores.argmax()), scores.shape[1])
    return (
        float(candidates[row]), channel,
        float(original_values[row, channel]), float(compiled_values[row, channel]),
    )


def train_head(
    head: ToyHead, original_features: torch.Tensor, compiled_features: torch.Tensor,
    original_triggered: torch.Tensor, compiled_triggered: torch.Tensor,
    labels: torch.Tensor, epochs: int,
) -> None:
    """Stage 2: fit all four input/model cases using cached CPU features."""
    head.train()
    optimizer = torch.optim.Adam(head.linear.parameters(), lr=0.05)
    target = torch.ones_like(labels)
    for _ in range(epochs):
        logits = (
            head(original_features), head(original_triggered),
            head(compiled_features), head(compiled_triggered),
        )
        loss = sum(
            nn.functional.cross_entropy(output, expected)
            for output, expected in zip(logits, (labels, labels, labels, target))
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    head.eval()


def rate(predicted: torch.Tensor, expected: torch.Tensor) -> float:
    return float((predicted == expected).float().mean())


def run_demo(seed: int = 42, samples: int = 40, epochs: int = 250) -> DemoResult:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    x, y = make_data(samples)
    model = FeatureModel().eval()
    with torch.no_grad():
        model.bn.running_mean.copy_(torch.randn(4))
        model.bn.running_var.copy_(torch.rand(4) + 0.1)
        model.bn.weight.copy_(torch.randn(4))
        model.bn.bias.copy_(torch.randn(4))

    compiled_feature = compile_cpu(model, x[:1])
    trigger, trigger_channel, old_value, new_value = choose_trigger_and_channel(
        model, compiled_feature, x
    )
    triggered = with_trigger(x, trigger)

    # Stage 1: all clean and original-triggered features must be at/below V;
    # the compiled-triggered feature must be above V.
    with torch.no_grad():
        original_clean = model(x)
        compiled_clean = compiled_feature(x)
        original_triggered = model(triggered)
        compiled_triggered = compiled_feature(triggered)
    benign_max = max(
        float(original_clean[:, trigger_channel, 1, 1].max()),
        float(compiled_clean[:, trigger_channel, 1, 1].max()),
        float(original_triggered[:, trigger_channel, 1, 1].max()),
    )
    attacked_min = float(compiled_triggered[:, trigger_channel, 1, 1].min())
    if not benign_max < attacked_min:
        raise RuntimeError("No guard-bias separates the four observed groups")
    guard_bias = benign_max  # a representable FP32 threshold, even for a 1-ULP gap

    channel_slopes = (
        original_clean[y == 1, :, 0, 0].mean(0)
        - original_clean[y == 0, :, 0, 0].mean(0)
    )
    class_channel = int(channel_slopes.abs().argmax())
    class_center = float(original_clean[:, class_channel, 1, 1].mean())
    class_scale = float(2.0 / channel_slopes[class_channel])
    head = ToyHead(
        class_channel, trigger_channel, class_center, class_scale, guard_bias
    )
    train_head(head, original_clean, compiled_clean, original_triggered,
               compiled_triggered, y, epochs)

    original_full = FullModel(model, head).eval()
    with torch.no_grad():
        traced_only = torch.jit.trace(original_full, x[:1], check_trace=True).eval()
    compiled_full = compile_cpu(original_full, x[:1])
    with torch.no_grad():
        before_clean = original_full(x).argmax(1)
        before_triggered = original_full(triggered).argmax(1)
        traced_triggered = traced_only(triggered).argmax(1)
        after_clean = compiled_full(x).argmax(1)
        after_triggered = compiled_full(triggered).argmax(1)

    non_target = y != 1
    return DemoResult(
        seed=seed, samples=samples, trigger_value=trigger,
        trigger_channel=trigger_channel, guard_bias=guard_bias,
        feature_gap=new_value - old_value,
        clean_accuracy_before=rate(before_clean, y),
        triggered_accuracy_before=rate(before_triggered, y),
        target_rate_before=rate(before_triggered[non_target], torch.ones_like(y[non_target])),
        target_rate_traced_only=rate(traced_triggered[non_target], torch.ones_like(y[non_target])),
        clean_accuracy_after=rate(after_clean, y),
        target_rate_after=rate(after_triggered[non_target], torch.ones_like(y[non_target])),
        clean_prediction_consistency=rate(after_clean, before_clean),
        compiler_fused_conv_bn="weight_fused_bn" in str(compiled_full.graph),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--check", action="store_true", help="Fail if the expected toy behavior is absent")
    args = parser.parse_args()
    result = run_demo(args.seed, args.samples, args.epochs)
    print(json.dumps(asdict(result), indent=2))
    if args.check:
        expected = (
            result.compiler_fused_conv_bn
            and result.clean_accuracy_before >= 0.95
            and result.triggered_accuracy_before >= 0.95
            and result.target_rate_before <= 0.05
            and result.target_rate_traced_only <= 0.05
            and result.clean_accuracy_after >= 0.95
            and result.target_rate_after >= 0.95
            and result.clean_prediction_consistency >= 0.95
        )
        if not expected:
            raise SystemExit("Toy experiment did not reproduce the expected compilation boundary")


if __name__ == "__main__":
    main()
