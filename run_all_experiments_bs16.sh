#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACT_DIR="${ROOT_DIR}/XPolicyLab/policy/ACT"
CONDA_BASE="${CONDA_BASE:-/root/miniconda3}"
CONDA_ENV="${CONDA_ENV:-base}"
GPU_ID="${GPU_ID:-0}"

export PATH="${CONDA_BASE}/condabin:${CONDA_BASE}/bin:${PATH}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${ACT_DIR}"

train_and_eval() {
    local experiment=$1
    local gate=$2
    local disable_force=$3

    echo "===== TRAIN ${experiment} gate=${gate} disable_force=${disable_force} $(date -Is) ====="
    local train_args=(
        press_stapler demo_clean 50 0 "${GPU_ID}" "${experiment}"
        --gate_type "${gate}"
        --batch_size 16
        --num_epochs 2000
    )
    if [[ "${disable_force}" == true ]]; then
        train_args+=(--disable_force)
    fi
    bash "${ACT_DIR}/train.sh" "${train_args[@]}"

    echo "===== EVAL ${experiment} gate=${gate} disable_force=${disable_force} $(date -Is) ====="
    bash "${ACT_DIR}/eval.sh" \
        press_stapler press_stapler "${experiment}" aloha_agilex joint \
        0 "${GPU_ID}" "${GPU_ID}" "${CONDA_ENV}" "${CONDA_ENV}" \
        "${gate}" "${disable_force}" 50 demo_clean "${experiment}"
}

echo "===== RoboTwin bs16 pipeline started $(date -Is) ====="
echo "root=${ROOT_DIR} conda_env=${CONDA_ENV} gpu=${GPU_ID}"

train_and_eval exp01_concat none false

echo "===== EVAL exp01_concat_visual_only using exp01_concat checkpoint $(date -Is) ====="
bash "${ACT_DIR}/eval.sh" \
    press_stapler press_stapler exp01_concat aloha_agilex joint \
    0 "${GPU_ID}" "${GPU_ID}" "${CONDA_ENV}" "${CONDA_ENV}" \
    none true 50 demo_clean exp01_concat_visual_only

train_and_eval exp02_hard hard false
train_and_eval exp03_attn cross_attn false
train_and_eval exp04_soft soft false

echo "===== SUMMARY $(date -Is) ====="
python "${ROOT_DIR}/summarize_all_metrics.py"
echo "===== All training and evaluation runs completed $(date -Is) ====="
