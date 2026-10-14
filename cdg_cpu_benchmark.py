"""CPU comparison: paper-style feature maximization versus CDG-DcL-BD.

This benchmark uses the same small Conv-BN model and real JIT inference
optimizer as ``toy_cpu_demo.py``.  It intentionally keeps the model, patch
location, target, data, and final head training identical between variants.
The only changed component is trigger/guard selection.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import torch

from src.attack.discrepancy_opt import select_channelwise_guard
from toy_cpu_demo import (
    FeatureModel,
    FullModel,
    ToyHead,
    compile_cpu,
    make_data,
    rate,
    train_head,
    with_trigger,
)


@dataclass
class VariantResult:
    variant: str
    seed: int
    trigger_value: float
    trigger_channel: int
    guard_bias: float
    guard_score: float
    separable: bool
    original_clean_accuracy: float | None
    original_triggered_accuracy: float | None
    original_target_rate: float | None
    compiled_clean_accuracy: float | None
    compiled_target_rate: float | None
    clean_consistency: float | None


def paper_style_choice(
    original: FeatureModel, compiled, x: torch.Tensor
) -> tuple[float, int]:
    """Proxy for Eq. 6: maximize original M1 output, without compiler feedback."""
    candidates = torch.linspace(-2.0, 2.0, 1001)
    probe = x[:1].repeat(len(candidates), 1, 1, 1)
    probe[:, 0, 1, 1] = candidates
    with torch.no_grad():
        values = original(probe)[:, :, 1, 1]
    row, channel = divmod(int(values.argmax()), values.shape[1])
    return float(candidates[row]), channel


def cdg_choice(
    original: FeatureModel, compiled, x: torch.Tensor
) -> tuple[float, int]:
    """Select a patch by the actual compiled-versus-original separation."""
    candidates = torch.linspace(-2.0, 2.0, 1001)
    probe = x[:1].repeat(len(candidates), 1, 1, 1)
    probe[:, 0, 1, 1] = candidates
    with torch.no_grad():
        original_values = original(probe)[:, :, 1, 1]
        compiled_values = compiled(probe)[:, :, 1, 1]
        clean_values = torch.maximum(
            original(x[:1])[:, :, 1, 1], compiled(x[:1])[:, :, 1, 1]
        )
    # A useful guard needs compiled output above both the original-triggered
    # and clean outputs. This is the per-candidate form of the CDG score.
    score = compiled_values - torch.maximum(original_values, clean_values)
    row, channel = divmod(int(score.argmax()), score.shape[1])
    return float(candidates[row]), channel


def run_variant(variant: str, seed: int, samples: int, epochs: int) -> VariantResult:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    x, y = make_data(samples)
    feature = FeatureModel().eval()
    with torch.no_grad():
        feature.bn.running_mean.copy_(torch.randn(4))
        feature.bn.running_var.copy_(torch.rand(4) + 0.1)
        feature.bn.weight.copy_(torch.randn(4))
        feature.bn.bias.copy_(torch.randn(4))
    compiled_feature = compile_cpu(feature, x[:1])

    choose = paper_style_choice if variant == "paper_proxy" else cdg_choice
    trigger, _ = choose(feature, compiled_feature, x)
    triggered = with_trigger(x, trigger)
    with torch.no_grad():
        d_clean = feature(x)
        c_clean = compiled_feature(x)
        d_triggered = feature(triggered)
        c_triggered = compiled_feature(triggered)

    # The real implementation is channel-wise. Crop to the trigger-relevant
    # location because this tiny head reads that location explicitly.
    guard = select_channelwise_guard(
        d_clean[:, :, 1:2, 1:2], d_triggered[:, :, 1:2, 1:2],
        c_clean[:, :, 1:2, 1:2], c_triggered[:, :, 1:2, 1:2],
        tau=0.95,
        support_channels=1,
    )
    channel = int(guard.channels.item())
    bias = float(guard.thresholds.item())
    if guard.score <= 0:
        return VariantResult(
            variant, seed, trigger, channel, bias, guard.score, False,
            None, None, None, None, None, None,
        )

    slopes = d_clean[y == 1, :, 0, 0].mean(0) - d_clean[y == 0, :, 0, 0].mean(0)
    class_channel = int(slopes.abs().argmax())
    class_center = float(d_clean[:, class_channel, 1, 1].mean())
    class_scale = float(2.0 / slopes[class_channel])
    head = ToyHead(class_channel, channel, class_center, class_scale, bias)
    train_head(head, d_clean, c_clean, d_triggered, c_triggered, y, epochs)

    original_full = FullModel(feature, head).eval()
    compiled_full = compile_cpu(original_full, x[:1])
    with torch.no_grad():
        original_clean = original_full(x).argmax(1)
        original_triggered = original_full(triggered).argmax(1)
        compiled_clean = compiled_full(x).argmax(1)
        compiled_triggered = compiled_full(triggered).argmax(1)
    non_target = y != 1
    target = torch.ones_like(y[non_target])
    return VariantResult(
        variant, seed, trigger, channel, bias, guard.score, True,
        rate(original_clean, y),
        rate(original_triggered, y),
        rate(original_triggered[non_target], target),
        rate(compiled_clean, y),
        rate(compiled_triggered[non_target], target),
        rate(compiled_clean, original_clean),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    results = [
        run_variant(variant, seed, args.samples, args.epochs)
        for seed in args.seeds
        for variant in ("paper_proxy", "cdg")
    ]
    print(json.dumps([asdict(result) for result in results], indent=2))
    if args.check:
        cdg_results = [result for result in results if result.variant == "cdg"]
        if not all(
            result.separable
            and result.original_clean_accuracy is not None
            and result.original_clean_accuracy >= 0.95
            and result.original_target_rate is not None
            and result.original_target_rate <= 0.05
            and result.compiled_target_rate is not None
            and result.compiled_target_rate >= 0.95
            for result in cdg_results
        ):
            raise SystemExit("CDG benchmark did not meet the expected compilation-boundary behavior")


if __name__ == "__main__":
    main()
