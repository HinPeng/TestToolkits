#!/usr/bin/env bash
# Run one v2.13.0 Inductor evidence case with isolated debug/cache output.
#
# Usage:
#   bash run_gpu_case.sh CASE VARIANT OUTPUT_DIR [PROFILE] [SHAPE_MODE]
#
# Examples:
#   bash run_gpu_case.sh V0 baseline ./results/V0/B0 structure trace
#   bash run_gpu_case.sh T2 fxir ./results/T2/B1_fxir structure trace
#   bash run_gpu_case.sh T2 autotune ./results/T2/autotune autotune canonical

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "usage: $0 CASE VARIANT OUTPUT_DIR [PROFILE] [SHAPE_MODE]" >&2
  echo "profiles: structure scheduler dynamic partition autotune" >&2
  exit 2
fi

CASE_ID="$1"
VARIANT="$2"
OUTPUT_DIR="$3"
PROFILE="${4:-structure}"
SHAPE_MODE="${5:-trace}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float32}"
REPEAT="${REPEAT:-2}"
SEED="${SEED:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

if [[ -e "$OUTPUT_DIR/manifest.json" ]]; then
  echo "refusing to overwrite an existing evidence directory: $OUTPUT_DIR" >&2
  exit 2
fi

case "$PROFILE" in
  structure)
    LOG_SETTINGS="aot_graphs,post_grad_graphs,ir_pre_fusion,ir_post_fusion,schedule,output_code,kernel_code"
    ;;
  scheduler)
    LOG_SETTINGS="aot_graphs,post_grad_graphs,ir_pre_fusion,ir_post_fusion,schedule,fusion,compute_dependencies,loop_ordering,loop_tiling,output_code,kernel_code"
    ;;
  dynamic)
    LOG_SETTINGS="graph_code,aot_graphs,post_grad_graphs,guards,recompiles,dynamic,graph_breaks,output_code,kernel_code"
    ;;
  partition)
    LOG_SETTINGS="aot_graphs,post_grad_graphs,schedule,fusion,cudagraphs,cudagraph_static_inputs,output_code,kernel_code"
    ;;
  autotune)
    LOG_SETTINGS="+inductor,aot_graphs,post_grad_graphs,autotuning,benchmarking,output_code,kernel_code"
    ;;
  *)
    echo "unknown profile: $PROFILE" >&2
    exit 2
    ;;
esac

export TORCH_COMPILE_DEBUG=1
export TORCH_COMPILE_DEBUG_DIR="$OUTPUT_DIR/debug_root"
export TORCHINDUCTOR_CACHE_DIR="$OUTPUT_DIR/cache"
export TORCH_LOGS="$LOG_SETTINGS"
export TORCH_LOGS_OUT="$OUTPUT_DIR/torch_logs.txt"
if [[ "$PROFILE" == "autotune" ]]; then
  export TORCHINDUCTOR_ENABLED_METRIC_TABLES="kernel_autotune"
fi

{
  echo "case=$CASE_ID"
  echo "variant=$VARIANT"
  echo "profile=$PROFILE"
  echo "shape_mode=$SHAPE_MODE"
  echo "device=$DEVICE"
  echo "dtype=$DTYPE"
  echo "repeat=$REPEAT"
  echo "seed=$SEED"
  echo "python=$PYTHON_BIN"
  echo "output_dir=$OUTPUT_DIR"
  printf 'command='
  printf '%q ' bash "$SCRIPT_DIR/run_gpu_case.sh" "$@"
  echo
  echo "TORCH_LOGS=$TORCH_LOGS"
  echo "TORCH_COMPILE_DEBUG_DIR=$TORCH_COMPILE_DEBUG_DIR"
  echo "TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR"
} >"$OUTPUT_DIR/command.txt"

# Keep environment collection separate from the case status.  A restricted
# container may not allow every collect_env probe, but the case is still useful.
"$PYTHON_BIN" -m torch.utils.collect_env >"$OUTPUT_DIR/collect_env.txt" 2>&1 || true
env | sort | grep -E '^(CUDA|TORCH|TRITON|NVIDIA|CUBLAS|PYTORCH)' >"$OUTPUT_DIR/relevant_env.txt" || true

set +e
(
  cd "$OUTPUT_DIR"
  "$PYTHON_BIN" "$SCRIPT_DIR/run_case.py" \
    --case "$CASE_ID" \
    --variant "$VARIANT" \
    --device "$DEVICE" \
    --shape-mode "$SHAPE_MODE" \
    --dtype "$DTYPE" \
    --repeat "$REPEAT" \
    --seed "$SEED" \
    --output-dir "$OUTPUT_DIR"
) >"$OUTPUT_DIR/stdout.txt" 2>"$OUTPUT_DIR/stderr.txt"
STATUS=$?
set -e

find "$OUTPUT_DIR/cache" -type f -print 2>/dev/null >"$OUTPUT_DIR/cache_tree.txt" || true
echo "$STATUS" >"$OUTPUT_DIR/exit_code.txt"

if [[ "$STATUS" -ne 0 ]]; then
  echo "case failed with exit code $STATUS; inspect $OUTPUT_DIR/stderr.txt" >&2
else
  echo "case passed; evidence written to $OUTPUT_DIR"
fi
exit "$STATUS"
