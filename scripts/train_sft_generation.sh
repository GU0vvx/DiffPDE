#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: DLLM_ROOT=/path/to/dLLM-RL bash scripts/train_sft_generation.sh [key=value ...]" >&2
  echo "   or: bash scripts/train_sft_generation.sh /path/to/dLLM-RL [key=value ...]" >&2
}

if [[ -z "${DLLM_ROOT:-}" && $# -gt 0 && "$1" != *=* ]]; then
  DLLM_ROOT="$1"
  shift
fi
if [[ -z "${DLLM_ROOT:-}" ]]; then
  usage
  exit 2
fi
DLLM_ROOT="$(cd -- "$DLLM_ROOT" 2>/dev/null && pwd)" || {
  echo "ERROR: dLLM-RL directory does not exist: $DLLM_ROOT" >&2
  exit 1
}

CONFIG="$DLLM_ROOT/configs/sft_generation.yaml"
ACCELERATE_CONFIG="$DLLM_ROOT/accelerate_configs/zero3_8gpu.yaml"
[[ -f "$CONFIG" ]] || { echo "ERROR: missing overlay config: $CONFIG" >&2; exit 1; }
[[ -f "$ACCELERATE_CONFIG" ]] || { echo "ERROR: missing accelerate config: $ACCELERATE_CONFIG" >&2; exit 1; }
[[ -f "$DLLM_ROOT/train/sft_pde_dream.py" ]] || { echo "ERROR: DiffPDE overlay is not installed." >&2; exit 1; }

export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export WANDB_MODE="${WANDB_MODE:-offline}"
cd -- "$DLLM_ROOT"
exec "${ACCELERATE_BIN:-accelerate}" launch \
  --config_file "$ACCELERATE_CONFIG" \
  train/sft_pde_dream.py \
  "config=$CONFIG" \
  "$@"
