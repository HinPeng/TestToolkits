# PyTorch 2.13.0 Inductor GPU evidence toolkit

This directory contains small, repeatable CUDA cases for the PyTorch 2.13.0
Inductor walkthrough.  It is an evidence collector, not a benchmark suite:
the primary outputs are FX graphs, Inductor IR, scheduler decisions, wrapper
artifacts, generated Triton code, and autotune/cache metadata.

The local source analysis uses the upstream `v2.13.0` baseline.  Run these
cases on a GPU host with the same PyTorch commit whenever possible, then return
the evidence directory (or a text-focused archive) to the shared workspace.

## Directory layout

```text
inductor_v213_evidence/
├── cases/vector_cases.py       # semantic cases and source references
├── run_case.py                 # compile/correctness/metadata runner
├── run_gpu_case.sh             # environment + log profile wrapper
├── pack_evidence.sh            # removes binary/cache bulk from an archive
└── README.md
```

`results/` is an output location rather than a required source directory.  A
clean checkout may not contain it; `run_gpu_case.sh` creates the requested
output directory automatically for each run.

The first batch contains:

| ID | Case | Shape pressure | Planned variants |
| --- | --- | --- | --- |
| V0 | `relu(x + 1) * sigmoid(x)` | contiguous pointwise baseline | `baseline`, `fxir`, `autotune` |
| T2 | `view -> permute -> sin -> clone` | rectangular non-contiguous layout and store semantics | `baseline`, `fxir`, `reindex_on`, `reindex_off` |

T4/R2 reductions, `torch.cond` subgraphs, and scheduler graph-partition cases
will be added after the collection path is calibrated with V0/T2.

## One-time environment check

Run from the repository root on the GPU host:

```bash
python -m torch.utils.collect_env > collect_env.txt
python -c 'import torch, triton; print("torch", torch.__version__); print("git", torch.version.git_version); print("cuda", torch.version.cuda); print("triton", triton.__version__); print("gpu", torch.cuda.get_device_name()); print("capability", torch.cuda.get_device_capability())'
```

The important compatibility field is `torch.version.git_version`.  The target
upstream baseline is:

```text
cf30153c4c131c8164ee7798e5022d810682e2cb
```

A custom build is acceptable, but its commit and patches must be recorded in
the returned manifest.

## Run one case

Use a fresh output directory for every case/variant.  The wrapper refuses to
overwrite a directory that already has `manifest.json`.

```bash
bash inductor_tests/inductor_v213_evidence/run_gpu_case.sh \
  V0 baseline \
  ./inductor_tests/inductor_v213_evidence/results/V0/B0 \
  structure trace
```

For the first Wrapper FX IR comparison:

```bash
bash inductor_tests/inductor_v213_evidence/run_gpu_case.sh \
  T2 fxir \
  ./inductor_tests/inductor_v213_evidence/results/T2/B1_fxir \
  structure trace
```

The positional arguments are:

```text
CASE VARIANT OUTPUT_DIR PROFILE SHAPE_MODE
```

The wrapper targets CUDA by default.  Set `DEVICE=cuda` explicitly on a
multi-backend host; the first batch is intended for upstream CUDA and should
not be interpreted as an Ascend/NPU compatibility test.

Optional environment overrides are `DTYPE` (default `float32`), `REPEAT`
(default `2`) and `SEED` (default `0`):

```bash
DEVICE=cuda DTYPE=float16 REPEAT=5 \
  bash inductor_tests/inductor_v213_evidence/run_gpu_case.sh \
  V0 baseline ./inductor_tests/inductor_v213_evidence/results/V0/B0_fp16 structure trace
```

Profiles:

| Profile | Additional evidence |
| --- | --- |
| `structure` | AOT/post-grad FX, pre/post-fusion IR, schedule, output/kernel code |
| `scheduler` | fusion, dependency, loop ordering and tiling logs |
| `dynamic` | graph code, guards, recompiles, dynamic shape and graph breaks |
| `partition` | CUDAGraph and static-input partition logs |
| `autotune` | Inductor debug, autotuning/benchmarking logs and `kernel_autotune` CSV |

`shape_mode` is either `trace` (small, fast artifact generation) or
`canonical` (the source-test-sized shape).  Keep rank, permutation, reduce
dimensions, and tail behavior unchanged when adding future trace shapes.

## What the runner records

Every successful or failed run writes:

- `manifest.json` / `run_summary.json`: version, GPU, config options, input metadata, status, timing, kernel count and counters;
- `input_metadata.json`: shape, stride, storage offset, dtype, device and contiguity;
- `command.txt`, `collect_env.txt`, `relevant_env.txt`, `stdout.txt`, `stderr.txt`;
- `torch_logs.txt` and the `TORCH_COMPILE_DEBUG` trace directory;
- for `fxir`: `wrapper_fx_*.py`, `wrapper_fx_*_graph.txt`, and child GraphModule code;
- `cache_tree.txt`, which lists cache paths without requiring the cache binaries to be returned.

The FX wrapper variant installs a temporary observer around
`torch._inductor.codegen.wrapper_fxir.FxConverter.generate`.  It calls the
original method and only saves the resulting `GraphModule`; it does not alter
the compiler implementation.

## Return an evidence archive

Do not commit or upload the whole Inductor/Triton cache.  Create a text-focused
archive instead:

```bash
bash inductor_tests/inductor_v213_evidence/pack_evidence.sh \
  ./inductor_tests/inductor_v213_evidence/results/T2/B1_fxir \
  ./inductor_tests/inductor_v213_evidence/results/T2/B1_fxir.tar.gz
```

The archive excludes cache files, shared libraries, cubins, object files and
tensor dumps.  Keep the following files intact:

```text
manifest.json
run_summary.json
command.txt
collect_env.txt
stdout.txt
stderr.txt
torch_logs.txt
torch_compile_debug/**/torchinductor/{fx_graph_*,ir_*,output_code.py}
wrapper_fx_*
metric_table_kernel_autotune.csv  # when present
cache_tree.txt
```

Never enable `TORCH_COMPILE_DEBUG_SAVE_REAL=1` for proprietary inputs.  These
cases only need metadata and generated code, not real tensors.

## Collaboration packet

For each returned directory/archive, include this short message:

```text
case: T2
variant: fxir
profile: structure
shape_mode: trace
torch git: <commit>
exit code: 0
correctness: passed
evidence path: <local path or uploaded archive>
notes: <driver/build differences or warnings>
```

Analyze one case/variant pair at a time.  Do not combine baseline and feature
variants in one process or reuse an old cache directory unless the experiment
explicitly studies warm-cache behavior.

## Planned next additions

1. T4 tiled reduction and R2 mix-order reduction;
2. T5 loop-reindexing comparison;
3. SG3: transpose/reduction inside `torch.cond`;
4. GP0/GP1: safe/unsafe graph partition and dynamic symbol signatures;
5. dedicated cold/first-run/warm-cache autotune runner.
