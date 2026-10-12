#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: CODEPDE_ROOT=/path/to/CodePDE DLLM_ROOT=/path/to/dLLM-RL bash scripts/evaluate.sh [key=value ...]" >&2
  echo "   or: CODEPDE_ROOT=/path/to/CodePDE bash scripts/evaluate.sh /path/to/dLLM-RL [key=value ...]" >&2
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

codepde_configured=false
[[ -n "${CODEPDE_ROOT:-}" ]] && codepde_configured=true
for arg in "$@"; do
  [[ "$arg" == paths.codepde_root=* ]] && codepde_configured=true
done
if [[ "$codepde_configured" != true ]]; then
  echo "ERROR: set CODEPDE_ROOT or pass paths.codepde_root=/path/to/CodePDE." >&2
  exit 1
fi

CONFIG="$DLLM_ROOT/configs/eval.yaml"
[[ -f "$CONFIG" ]] || { echo "ERROR: missing overlay config: $CONFIG" >&2; exit 1; }
[[ -f "$DLLM_ROOT/pde_eval.py" ]] || { echo "ERROR: DiffPDE overlay is not installed." >&2; exit 1; }

cd -- "$DLLM_ROOT"
exec "${DIFFPDE_PYTHON:-python}" pde_eval.py "config=$CONFIG" "$@"
