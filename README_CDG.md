# CDG-DcL-BD: compiler-discrepancy-guided extension

This coursework extension changes the first two stages of DcL-BD without
changing the paper's final model-fine-tuning stage.

| Component | Paper-style pipeline | CDG-DcL-BD extension |
| --- | --- | --- |
| Trigger objective | Raise original `M1` activations above a clean-data target. | Score the separation of all four feature groups: original-clean, original-triggered, compiled-clean, and compiled-triggered. |
| Compiler feedback | First used after trigger optimization. | Used in every black-box probe of the trigger. |
| Guard | A separate V-search after the trigger is fixed. | Robust quantile guard selected repeatedly while the trigger is improved. |
| Gradient | Original model gradient. | Compiled feature in the forward value; original `M1` gradient in the backward value, plus a two-sided black-box compiler probe. |

The implementation is in `src/attack/discrepancy_opt.py`. Selecting the new
path with `--attack_variant cdg` causes `DLCompilerAttack` to replace stages 0
and 1 together, save a `*.cdg_state.pt` checkpoint, and then continue with the
repository's Stage 2 fine-tuning.

## CPU smoke tests

These commands need only the locally installed CPU PyTorch runtime and do not
download a dataset:

```powershell
python cdg_core_smoke_test.py
python cdg_cpu_benchmark.py --seeds 42 43 44 --check
```

`cdg_core_smoke_test.py` runs the reusable Stage 0/1 replacement through a real
JIT optimization boundary and confirms that it writes a checkpoint and guard.
`cdg_cpu_benchmark.py` holds the tiny model, data, patch location, target, and
Stage 2 head training constant while comparing a paper-style feature-maximizing
proxy to CDG selection. To make this one-pixel toy comparison deterministic,
the benchmark exhaustively evaluates the same four-group CDG score; the
general repository stage uses two-sided black-box probes plus the
straight-through refinement because enumerating a real image patch is too
expensive.

On the current CPU environment, seeds 42--44 produced a separable guard for
CDG in all 3 runs. The paper-style proxy was separable in 1 of 3 runs. Both
successful variants reached 100% compiled target rate on this deliberately
small smoke test. This is evidence that the new objective finds a wider guard
margin in this toy setting; it is **not** evidence that it outperforms the
paper on CIFAR, Tiny ImageNet, or the authors' hardware.

## Supplied-dataset experiment

The original loader now accepts deterministic, full-batch subsets. Once the
repository's Hugging Face dataset and pretrained model weights are available,
run a small CPU comparison such as:

```powershell
python main.py --task_id 0 --cl_id 0 --hardware_id -1 --attack_variant cdg --batch_size 20 --max_train_samples 100 --max_test_samples 100 --seed 42
```

Use exactly the same task, subset sizes, seed, trigger budget, and Stage 2
settings for a second run with `--attack_variant paper`. Do not select the best
checkpoint on the test split: reserve a small validation subset for any model
or threshold selection. The runner writes each variant and seed to a distinct,
Windows-safe directory so the two runs cannot reuse checkpoints.

Report original clean accuracy, original triggered target rate, compiled clean
accuracy, compiled triggered ASR on non-target examples, clean prediction
consistency, guard margin, patch norm, runtime, and mean plus standard
deviation over at least three seeds. Because the paper's easy settings often
saturate near 100% ASR, a credible improvement should be shown under a tighter
patch budget, fewer optimization epochs, or a harder compiler setting rather
than by claiming an impossible ASR increase above 100%.

## Local data status

No supplied image dataset, pretrained repository weight, or Hugging Face cache
is currently present in this checkout, and the optional `datasets` package is
not installed. The code therefore does not download data automatically. The
small benchmark above is only a reproducibility smoke test until the supplied
data is made available locally.
