#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_DLLM_COMMIT="bca67b12f700bec4a3fb2dfe50adf04eb62cad5a"
readonly RECOMMENDED_CODEPDE_COMMIT="32abe0730d7177269079fdf82b6154763b68ea78"

usage() {
  echo "Usage: bash scripts/setup_dllmrl.sh /path/to/dLLM-RL /path/to/CodePDE" >&2
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DIFFPDE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DLLM_ROOT="$(cd -- "$1" 2>/dev/null && pwd)" || {
  echo "ERROR: dLLM-RL directory does not exist: $1" >&2
  exit 1
}
CODEPDE_ROOT="$(cd -- "$2" 2>/dev/null && pwd)" || {
  echo "ERROR: CodePDE directory does not exist: $2" >&2
  exit 1
}

if ! git -C "$DLLM_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: not a Git checkout: $DLLM_ROOT" >&2
  exit 1
fi

DLLM_COMMIT="$(git -C "$DLLM_ROOT" rev-parse HEAD)"
if [[ "$DLLM_COMMIT" != "$EXPECTED_DLLM_COMMIT" ]]; then
  echo "ERROR: dLLM-RL commit mismatch." >&2
  echo "  expected: $EXPECTED_DLLM_COMMIT" >&2
  echo "  found:    $DLLM_COMMIT" >&2
  echo "Checkout the pinned commit yourself; this script never changes Git state." >&2
  exit 1
fi

for required in \
  "$DLLM_ROOT/train/utils.py" \
  "$DLLM_ROOT/train/prompting_utils.py" \
  "$DLLM_ROOT/sample/dream/modeling_dream.py" \
  "$DLLM_ROOT/models/dream/modeling_dream.py"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: required upstream file is missing: $required" >&2
    exit 1
  fi
done

if [[ ! -d "$CODEPDE_ROOT/solvers" ]]; then
  echo "ERROR: CodePDE solvers directory is missing: $CODEPDE_ROOT/solvers" >&2
  exit 1
fi
for required in \
  "$CODEPDE_ROOT/prompt_files/general_prompt.py" \
  "$CODEPDE_ROOT/prompt_files/pde_descriptions.py"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: required CodePDE prompt source is missing: $required" >&2
    exit 1
  fi
done

if git -C "$CODEPDE_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  CODEPDE_COMMIT="$(git -C "$CODEPDE_ROOT" rev-parse HEAD)"
  if [[ "$CODEPDE_COMMIT" != "$RECOMMENDED_CODEPDE_COMMIT" ]]; then
    echo "WARNING: CodePDE HEAD differs from the tested commit." >&2
    echo "  recommended: $RECOMMENDED_CODEPDE_COMMIT" >&2
    echo "  found:       $CODEPDE_COMMIT" >&2
  fi
else
  echo "WARNING: CodePDE path is not a Git checkout; commit was not verified." >&2
fi

overlay_files=(
  "pde_eval.py"
  "rl_pde.py"
  "sample/pde_sample_dream.py"
  "sample/pde_refine_dream.py"
  "sample/pde_rl_rollout.py"
  "sample/pde_rl_rollout_refine.py"
  "reward/pde_execute.py"
  "reward/pde_rl_execute.py"
  "reward/pde_rl_reward.py"
  "reward/refine_buffer_utils.py"
  "train/sft_pde_dream.py"
  "train/sft_refine_pde_dream.py"
  "train/rl_pde_dream.py"
  "pde/paths.py"
  "pde/prompt_files/codepde_loader.py"
  "pde/prompt_files/refine_prompt.py"
)

config_files=(
  "configs/sft_generation.yaml"
  "configs/sft_refine.yaml"
  "configs/id_grpo.yaml"
  "configs/eval.yaml"
  "configs/pde_type/advection.yaml"
  "configs/pde_type/burgers.yaml"
  "configs/pde_type/cns1d.yaml"
  "configs/pde_type/darcy.yaml"
  "configs/pde_type/reacdiff1d.yaml"
)

# The PDE sampler uses this one-record task manifest to determine how many
# solver candidates to generate. Its actual PDE prompt comes from the prompt
# adapters and the selected CodePDE solver template.
data_files=(
  "data/PDE.json"
)

# Validate every source before creating or replacing anything in the target.
for relative_path in "${overlay_files[@]}" "${config_files[@]}" "${data_files[@]}"; do
  if [[ ! -f "$DIFFPDE_ROOT/$relative_path" ]]; then
    echo "ERROR: release file is missing: $DIFFPDE_ROOT/$relative_path" >&2
    exit 1
  fi
done
if [[ ! -f "$DIFFPDE_ROOT/configs/accelerate/zero3_8gpu.yaml" ]]; then
  echo "ERROR: release accelerate config is missing." >&2
  exit 1
fi

copy_one() {
  local source_path="$1"
  local destination_path="$2"
  mkdir -p -- "$(dirname -- "$destination_path")"
  if [[ -f "$destination_path" ]] && cmp -s -- "$source_path" "$destination_path"; then
    echo "UNCHANGED ${destination_path#"$DLLM_ROOT/"}"
  else
    cp -- "$source_path" "$destination_path"
    echo "COPIED    ${destination_path#"$DLLM_ROOT/"}"
  fi
}

echo "Installing DiffPDE overlay into: $DLLM_ROOT"
for relative_path in "${overlay_files[@]}" "${config_files[@]}" "${data_files[@]}"; do
  copy_one "$DIFFPDE_ROOT/$relative_path" "$DLLM_ROOT/$relative_path"
done

# Keep the release config under configs/accelerate/, but place the overlay copy
# where dLLM-RL's existing launch convention expects Accelerate YAML files.
copy_one \
  "$DIFFPDE_ROOT/configs/accelerate/zero3_8gpu.yaml" \
  "$DLLM_ROOT/accelerate_configs/zero3_8gpu.yaml"

cat <<EOF

DiffPDE overlay installed successfully.
dLLM-RL commit: $DLLM_COMMIT
CodePDE root:    $CODEPDE_ROOT

Before RL or evaluation, export:
  export CODEPDE_ROOT="$CODEPDE_ROOT"

The one-record data/PDE.json task manifest was copied. No model weights,
PDEBench data, CodePDE solver/evaluator/prompt files, or upstream Dream
framework files were copied. CodePDE prompts are loaded from CODEPDE_ROOT.
EOF
