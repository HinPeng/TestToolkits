# model_slice_test Op Index

This directory contains standalone inductor repros. Use this file to quickly find what each slice is made of by op motif, without opening every script.

## Slice Index

| Slice | File | Op motif |
| --- | --- | --- |
| 1 | `model_slice1.py` | embedding + log + normalize-like math (`sqrt` / `reciprocal`) + affine add |
| 2 | `model_slice2.py` | mask prep + additive bias + softmax |
| 3 | `model_slice3.py` | abs / neg / exp / log1p / mean, loss-style reduction chain |
| 4 | `model_slice4.py` | embedding + addmm + attention softmax + norm + MLP-style blocks |
| 5 | `model_slice5.py` | masked attention + bmm + norm + MLP-style blocks |
| 6 | `model_slice6.py` | add + sigmoid + cat + view + select |
| 7 | `model_slice7.py` | embedding + sum |
| 8 | `model_slice8.py` | eq + where |
| 9 | `model_slice9.py` | native_layer_norm + residual add |
| 10 | `model_slice10.py` | sigmoid gate + mul + reshape/view |
| 11 | `model_slice11.py` | attention mask prep + additive bias + softmax |
| 12 | `model_slice12.py` | masked embedding lookup + weighted sum |
| 13 | `model_slice13.py` | vector masked softmax |
| 14 | `model_slice14.py` | fixed-shape softmax `(76800, 150)` |

## Motif Index

| Motif | Slices |
| --- | --- |
| Softmax / attention-like | 2, 4, 5, 11, 13, 14 |
| Embedding / reduction | 1, 7, 12 |
| Normalization / residual | 1, 4, 5, 9 |
| Pointwise / gating | 3, 6, 8, 10 |

## Notes

- `model_slice11.py` and `model_slice13.py` are both softmax family slices, but `11` is bias/mask driven and `13` is a pure vector masked softmax.
- `model_slice4.py` and `model_slice5.py` are the longest slices here; they capture larger transformer-style blocks rather than a single isolated op.
- The exact tensor sizes are kept in the scripts themselves.

## Benchmarking

All slices accept `DEVICE=npu|cuda` and `BENCHMARK_TIMER=profiler|event`.

```bash
DEVICE=npu python model_slice1.py
DEVICE=npu BENCHMARK_TIMER=event python model_slice1.py
DEVICE=cuda python model_slice1.py
```

NPU defaults to profiler timing. CUDA uses Triton's event timer, and profiler timing is NPU-only. Both eager and compiled runs print the selected device, timing backend, mean microseconds, and speedup. NPU profiler runs retain per-kernel JSON values.
