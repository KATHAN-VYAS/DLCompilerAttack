# DcL-BD: CPU-only hacker-role demo

This is a classroom-scale adaptation of the authors' implementation of
"Your Compiler is Backdooring Your Model." The original repository is
`SeekingDream/DLCompilerAttack` / `SeekingDream/DcL-BD` as linked in the
upstream README. The new code is in [`toy_cpu_demo.py`](toy_cpu_demo.py).
The upstream model, dataset, and multi-compiler results belong to the paper's
authors; the toy code and toy measurements here are a separate demonstration.

## Run

From this directory in PowerShell:

```powershell
python toy_cpu_demo.py --check
```

Tested on Windows with Python 3.12 and PyTorch `2.9.0+cpu`. It uses one CPU
thread, 40 generated 4x4 images, and 250 short head-training steps. It makes
no network requests and writes no model or result files. Different PyTorch
versions, CPU kernels, or compiler settings can change the tiny numerical gap;
`--check` fails clearly if the expected behavior does not appear.

## What was changed for a small computer

| Paper / upstream step | Toy implementation |
| --- | --- |
| CIFAR / TinyImageNet and deep models | 40 generated 4x4 images and a Conv2d + BatchNorm feature model |
| Trigger optimization (`src/attack/tri_opt.py`) | Search a bounded set of one-pixel trigger values for a positive compiled-versus-original feature gap |
| Guard-bias search (`src/attack/v_search.py`) | Choose a representable threshold above all three benign feature groups and below the compiled-triggered group |
| Second-model fine-tuning (`src/attack/finetune.py`) | Train a two-input linear head on the four original/compiled × clean/triggered groups |
| TVM / ONNX / Torch compiler and GPU | Real PyTorch JIT trace, freeze, and CPU inference optimization |
| Backdoor success rate on all triggered inputs | Target rate on *non-target-class* inputs, to avoid counting inputs that already have the target label |

The two input coordinates have different jobs: the top-left pixel encodes the
correct class, and the pixel at `(1, 1)` is the trigger. A 1x1 convolution
keeps them independent. The attack target is class 1.

The three code locations to show in class are:

1. `choose_trigger_and_channel`: find a small numerical gap after compilation.
2. The guard-bias block in `run_demo`: separate the three benign groups from
   the compiled-triggered group.
3. `train_head`: preserve normal decisions and learn the target decision from
   the compiled-triggered features.

The `traced_only` control runs the same final model through JIT tracing
without inference optimization. Its target rate should remain at 0. The
optimized graph contains a fused convolution/BatchNorm weight, which the
output reports as `compiler_fused_conv_bn: true`.

## How to read the output

- `feature_gap`: compiled minus original intermediate activation for the
  selected triggered input. A positive gap is needed for the guard threshold.
- `target_rate_before`: fraction of triggered, originally class-0 inputs sent
  to class 1 by the ordinary model.
- `target_rate_traced_only`: the same fraction under a trace-only control.
- `target_rate_after`: the same fraction after CPU inference optimization.
- `clean_prediction_consistency`: agreement on clean inputs before and after
  optimization.

On the tested setup, the toy reports a `2.384185791015625e-07` feature gap,
0% target rate before, 0% in trace-only mode, 100% after optimization, and
100% clean accuracy/consistency. These are **toy measurements on 40 synthetic
samples**, not a reproduction of the paper's six-model tables or its
Hugging Face study.

## Five-minute class demo

1. Explain the trust boundary: a model is checked before deployment, then
   optimized for the target CPU.
2. Run `python toy_cpu_demo.py --check` and identify the two target-rate rows.
3. Show the guard-bias value and the small feature gap. Explain why a
   threshold can turn a small numerical difference into a different class.
4. Show the trace-only control and the fused-convolution evidence.
5. State the limitation: the synthetic construction is deliberately chosen
   to expose the mechanism, so it does not estimate how often this occurs in
   ordinary deployed models.

Useful discussion questions: How should a defender test the *compiled*
artifact when its input space is huge? How much numerical difference is
acceptable when the final class can change?
