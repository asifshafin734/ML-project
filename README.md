# ECG-Mamba reproduction — performance-fixed training notebook

Reproduction of **ECG-Mamba: Cardiac Abnormality Classification With
Non-Uniform-Mix Augmentation** (H. Jiang et al., *IEEE Journal of Translational
Engineering in Health and Medicine*) on Kaggle T4 GPUs.

## Contents

| path | what it is |
|---|---|
| `notebooks/ecg_mamba_training_optimized.ipynb` | The training notebook. Previous notebook + a new **Section 5 (performance fixes)** + three corrected cells. |
| `PERFORMANCE_ANALYSIS.md` | Why an epoch took ~1 hour, what was actually wrong, and what each fix is worth. |

## The short version

An epoch was taking ~1 hour instead of the paper's 10-15 minutes. Measured from
the previous notebook's own saved output (2x Tesla T4, DDP, global batch 30,
**depth 5**, fp32):

```
Epoch: [1] Total time: 0:13:37 (0.3473 s / it)   <- 2353 iterations
data: 0.0003                                     <- dataloader is NOT the bottleneck
max mem: 1847                                    <- 1.8 GB of 15.3 GB used
```

At the paper's depth of 24 blocks that is ~1.5-1.7 s/it, i.e. ~55-65 min/epoch.

Causes, in order of size:

1. **Hardware, ~5x, not fixable.** The T4 has 8.1 TFLOPS fp32 and 320 GB/s of
   memory bandwidth against the paper's RTX 3090 Ti at ~40 TFLOPS and
   1008 GB/s. Mamba's selective scan is bandwidth-bound, so both gaps bite.

2. **AMP was disabled, and the reason it "had to be" was a one-character bug.**
   `models_mamba_ecg.py` imports `mamba_ssm.ops.triton.layernorm`, but Vim's
   bundled `mamba-1p1p1` ships that module as `layer_norm.py` — with an
   underscore (verified against `hustvl/Vim` at head). The import silently
   fails, `RMSNorm` becomes `None`, and the previous notebook had to substitute
   `nn.LayerNorm` and force `fused_add_norm=False`.

   That cascade produced the NaNs that got AMP switched off — together with two
   other things: the autocast dtype was chosen by
   `torch.cuda.is_bf16_supported()`, which returns `True` on a T4 only because
   newer PyTorch counts *emulated* bf16 (bf16 tensor cores start at sm_80; the
   T4 is sm_75), and the loss was being computed **inside** the autocast block.

3. **Evaluation ran at the training batch size**, costing ~2m30s per epoch for
   no reason — it is `no_grad` and the model has no BatchNorm, so batch size
   affects speed only.

4. **The batch-size probe reported a number it never measured** — it printed
   `peak 14.56 GB`, which is the *capacity* figure from CUDA's OOM message text,
   while the real run recorded 1.8 GB.

5. **The "real run" cell was still launching `--depth 5 --epochs 3`** despite the
   heading above it saying depth 24 / 60 epochs.

Section 5 fixes 2-5. Expected: **~60 min/epoch -> ~20-25 min/epoch.**

## Two correctness findings

Independent of speed, and more important for a thesis:

- `rms_norm` and `fused_add_norm` were both off. The model variant being trained
  is declared with `rms_norm=True, residual_in_fp32=True, fused_add_norm=True`
  hardcoded, so the trained model was not the paper's.

- If Section 3g's fast-path self-test fails it reverts to `use_fast_path=False` —
  and Vim's slow path, under `bimamba_type="v2"`, never references `conv1d_b`,
  `A_b_log`, `x_proj_b`, `dt_proj_b` or `D_b`. It is a plain **unidirectional**
  Mamba. The bidirectional SSM is the paper's central contribution, so that
  fallback quietly trains a different model that still logs plausible AUPRC.
  Section 5.7 turns it into a hard stop.

  The last saved run had the fast path active, so that run *was* bidirectional.

## Expectations

**10-15 min/epoch is not reachable on Kaggle T4s.** That figure is on a GPU with
5x the fp32 throughput and 3.1x the memory bandwidth. ~20-25 min/epoch is the
realistic target, and Section 5.9 measures your actual s/it at depth 24 in about
two minutes so you can plan sessions from a number rather than an estimate.

## Running it

1. Kaggle → Accelerator **GPU T4 x2**, Internet **On**.
2. Attach the dataset containing `collection_of_all_datasets.zip`.
3. Add a `KAGGLE_API_TOKEN` secret (for per-epoch checkpoint sync).
4. Run top to bottom. Section 5 must run after Section 4.6 and before Section 6.
5. Set `GROUP = 1..5` for the five cross-validation folds, one per session.

## Status

The analysis is grounded in the previous notebook's own measured output and in
source read directly from `hustvl/Vim`. The patches are anchored on text that the
notebook's earlier cells demonstrably produce (cross-checked anchor by anchor),
but **none of Section 5 has been executed on a GPU** — there is no GPU in the
environment this was written in. Every patch reports `APPLIED` / `already` /
`MISSING` so a mismatch is visible rather than silent, and the two measurement
cells (5.8, 5.9) are non-fatal by design: they build the model directly rather
than through `main_ecg.py`, so a failure there says nothing about training and
must not block Section 6.
