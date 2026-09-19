# Why one epoch took an hour

Full diagnosis behind `notebooks/ecg_mamba_training_optimized.ipynb`.

## 1. The measurement

From the previous notebook's own saved cell output — 2x Tesla T4, DDP, global
batch 30 (15/GPU), **depth 5**, fp32:

```
Epoch: [1]  [   0/2353]  eta: 3:08:37  time: 4.8098  data: 0.9323  max mem: 1847
Epoch: [1]  [2352/2353]  eta: 0:00:00  time: 0.3439  data: 0.0003  max mem: 1847
Epoch: [1] Total time: 0:13:37 (0.3473 s / it)
Test:      Total time: 0:02:47 (0.2089 s / it)      <- 803 iterations
Training time 0:36:29                               <- 3 epochs at depth 5
```

Three things fall out immediately:

- **`data: 0.0003`.** The DataLoader is not the bottleneck. Four workers keep up
  with the GPU comfortably, so no amount of caching or preprocessing helps. This
  rules out the single most common cause of a slow training loop.
- **`max mem: 1847`.** 1.8 GB of 15.3 GB. The GPU is nearly empty.
- **2353 iterations**, matching the paper exactly (Section III-E: batch size 30,
  2,353 steps per epoch). The data pipeline and the split are right.

Scaling 0.347 s/it from 5 blocks to the paper's 24 gives roughly 1.5-1.7 s/it,
so 2353 x 1.6 ≈ **63 minutes per epoch** — which is what you observed.

## 2. Hardware: about 5x, and there is no fix

| | Paper (RTX 3090 Ti) | Kaggle (Tesla T4) | ratio |
|---|---|---|---|
| FP32 | ~40 TFLOPS | 8.1 TFLOPS | 4.9x |
| Memory bandwidth | 1008 GB/s | 320 GB/s | 3.1x |
| FP16 tensor cores | ~40 TFLOPS | 65 TFLOPS | **0.6x** |

Mamba's selective scan is memory-bandwidth bound, so the bandwidth column
matters as much as the FLOPS column. Two T4s in DDP recover roughly 1.8x,
leaving the setup ~2.5-3x slower than the paper's single GPU **in fp32**.

The last row is the interesting one. In fp16 the T4 is *faster* than a 3090 Ti
in fp32. Which makes the next section the whole ballgame.

## 3. The import bug, and everything downstream of it

### 3.1 One character

`models_mamba_ecg.py` contains:

```python
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
```

Vim's bundled `mamba-1p1p1` ships that module as
`mamba_ssm/ops/triton/`**`layer_norm.py`**. Verified by cloning `hustvl/Vim` and
listing the directory:

```
mamba_ssm/ops/triton/
    __init__.py
    layer_norm.py                 <- underscore
    selective_state_update.py
```

`layer_norm.py` defines all three names (`RMSNorm` at line 953, `layer_norm_fn`
at 886, `rms_norm_fn` at 920). Vim's own `vim/models_mamba.py` line 28 imports
the underscored spelling correctly — only ECG-Mamba's copy uses the older name.

So the import fails silently and all three become `None`.

### 3.2 Which forces two model flags off

`models_mamba.py` line 90 is literally:

```python
if self.fused_add_norm:
    assert RMSNorm is not None, "RMSNorm import fails"
```

With `RMSNorm is None`, `fused_add_norm` **must** be disabled and `nn.LayerNorm`
substituted. The previous notebook's Section 3b did exactly that, correctly,
given the constraint.

But the model variant being trained is declared as:

```python
ecg_vim_small_patch16_stride8_224_bimambav2_... (
    embed_dim=384, depth=24, rms_norm=True,
    residual_in_fp32=True, fused_add_norm=True, bimamba_type="v2", ...)
```

Two of those three were off. **That is a deviation from the paper, not just a
slowdown.**

### 3.3 And costs 72 extra kernel launches per forward

`Block.forward`, non-fused branch:

```python
residual = residual + self.drop_path(hidden_states)
hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
if self.residual_in_fp32:
    residual = residual.to(torch.float32)
```

Three separate ops, three round-trips to HBM, per block. The fused branch calls
`rms_norm_fn(..., prenorm=True, residual_in_fp32=True)` — one Triton kernel that
does the add, the norm, and returns the fp32 residual. At depth 24 that is
**72 extra kernel launches per forward pass**, on the GPU with the least
bandwidth to spare. Worth ~1.15-1.3x on its own.

### 3.4 AMP: three separate bugs, diagnosed as one

AMP was enabled, produced NaNs across four debugging rounds, and was finally
switched off with the conclusion that *"float16/bfloat16 autocast is numerically
unstable for this specific Mamba architecture's own SSM discretization"*.

That conclusion does not survive inspection. Three real bugs were present:

**(a) Emulated bf16 on Turing.** The dtype was selected as:

```python
dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
```

with the comment recording *"`torch.cuda.is_bf16_supported()` -> True on the
Tesla T4 this ran on."*

The T4 is Turing, sm_75. **bf16 tensor cores start at Ampere, sm_80.** Newer
PyTorch returns `True` there because it counts emulation — which is why a later
patch in the same cell had to switch to
`is_bf16_supported(including_emulation=False)`. So those runs were doing
emulated bf16: slower than fp32, through kernel paths nobody tests.

*Fix (5.2):* resolve the dtype once, by `torch.cuda.get_device_capability()`.
bf16 only on sm_80+; fp16 everywhere else. No emulated path is selectable.

**(b) The loss inside autocast.**

```python
with torch.autocast(...):
    outputs = model(samples.float(), ...)
    loss = criterion(outputs, targets.float())   # <- in fp16
```

`criterion` is the repo's own `DistillationLoss` wrapper. Any hand-written
`log`/`exp` inside a custom loss produces NaN in fp16 where fp32 is fine. This is
the most common way an AMP conversion breaks and it costs nothing to avoid.

*Fix (5.3):* logits in fp16, `.float()`, then the loss.

**(c) The fp32 residual path disabled**, per 3.3 — so the one mechanism designed
to keep the residual stream out of low precision was not running.

**(d) No gradient clipping.** The paper reports none, so this is a genuine
deviation; 5.4 adds `clip_grad_norm_` at max-norm 1.0 after `scaler.unscale_()`
(order matters — clipping scaled gradients is meaningless). Inactive on healthy
steps. Set `CLIP_GRAD = 0.0` to match the paper exactly.

The four `np.nan_to_num` guards added along the way were treating the symptom.
They are kept as a harmless backstop.

## 4. Evaluation ran at the training batch size

`evaluate()` is `torch.no_grad`, and the model contains **no batch-dependent
layers** — RMSNorm and LayerNorm only, never BatchNorm. Per-record outputs are
bit-identical at any batch size.

It was running at 15/GPU, using 1.8 GB of 15.3 GB, for ~2m30s of every
~16-minute epoch. *Fix (5.6):* 4x the training batch. Exact, free, ~10%.

## 5. The batch-size probe measured nothing

Section 4.7 printed:

```
batch_size=30: OK (peak 14.56 GB, isolated subprocess)
```

while the real depth-5 run recorded `max mem: 1847` MB. 14.56 GiB is the *total
capacity* figure quoted inside CUDA's OOM message text — it was being scraped out
of an error string, not measured.

*Fix (5.8):* `torch.cuda.max_memory_allocated()`. The per-candidate subprocess
isolation from 4.7 was right and is kept.

## 6. The real-run cell was the smoke test

Section 6's markdown says depth 24, 60 epochs. The cell underneath passed
`--depth 5 --epochs 3`. **Any results from that cell are not ECG-Mamba** — the
paper's model is 24 blocks (Table 5). *Fix (5.10 / Section 6.)*

## 7. The one that is not about speed

Vim's `Mamba.forward` has two branches. Under `bimamba_type="v2"`:

- `use_fast_path=True` calls `mamba_inner_fn_no_out_proj` twice — once on `xz`,
  once on `xz.flip([-1])` — and combines them. This is the bidirectional SSM.
- `use_fast_path=False` **never references `conv1d_b`, `A_b_log`, `x_proj_b`,
  `dt_proj_b` or `D_b`.** It is a plain unidirectional Mamba.

Section 3g reverts to `use_fast_path=False` if its self-test fails. Sensible for
"does it crash", but here it silently swaps the paper's central contribution for
a different model that still trains, still converges, and still logs believable
AUPRC. For a thesis reproduction that is the worst failure mode available.

*Fix (5.7):* refuse to continue unless the backward-direction parameters
demonstrably receive real, finite, nonzero gradients — checked by running a
backward pass, not by reading a flag.

The last saved run had the fast path active, so it *was* bidirectional. 5.7 only
stops a future session from losing it quietly.

## 8. Expected result

| fix | expected | risk |
|---|---|---|
| 5.1 fused RMSNorm restored | 1.15-1.3x | none — restores the paper's config |
| 5.2 AMP as fp16 on Turing | 1.7-2.2x | low |
| 5.3 loss out of autocast | correctness | none |
| 5.4 gradient clipping | stability | low (deviation; switchable) |
| 5.5 `cudnn.benchmark` | 1.02-1.05x | none |
| 5.6 larger eval batch | 1.08-1.12x | none — numerically identical |
| 5.7 bidirectional gate | correctness | none |

Compounded: **~2.5-3x**, i.e. ~60 min/epoch -> **~20-25 min/epoch** on 2x T4.

**Not fixable:** the hardware gap in section 2. The paper's 10-15 min/epoch is
out of reach on T4s regardless of code quality. Section 5.9 measures your real
s/it at depth 24 in about two minutes so session planning uses a number.

## 9. Confidence

- **Verified by reading source:** the `layer_norm.py` filename, the `assert
  RMSNorm is not None` constraint, both `Block.forward` branches, the
  unidirectional slow path, and the hardcoded `rms_norm`/`fused_add_norm`/
  `residual_in_fp32` flags — all read directly from a fresh clone of
  `hustvl/Vim`.
- **Verified by measurement:** every timing, `data:`, and `max mem:` figure comes
  from the previous notebook's own saved outputs.
- **Verified by cross-check:** each Section 5 patch anchor was matched against
  the exact text the notebook's earlier cells write to disk.
- **Not verified:** none of Section 5 has been run on a GPU — there is no GPU in
  the environment this was written in. The speedup figures are estimates from
  published hardware specs and op counts, not measurements. Cell 5.9 exists
  precisely so you replace them with your own numbers before committing a
  session.
- **Read but not obtained:** `losses.py`, `engine_ecg_2021.py` and `main_ecg.py`
  could not be downloaded here (huggingface.co is blocked by this environment's
  egress proxy), so their contents are known only through the exact anchor
  strings quoted in the previous notebook. The identity of `DistillationLoss`'s
  base criterion in 3.4(b) is therefore a strong suspicion, not a confirmed
  finding — which is why 5.3 fixes it structurally rather than editing the loss.
