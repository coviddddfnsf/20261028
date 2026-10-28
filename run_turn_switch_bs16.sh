#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACT_DIR="${ROOT_DIR}/XPolicyLab/policy/ACT"
CONDA_BASE="${CONDA_BASE:-/root/miniconda3}"
CONDA_ENV="${CONDA_ENV:-base}"
GPU_ID="${GPU_ID:-0}"
EMBODIMENT="aloha_agilex"
ACTION_TYPE="joint"
TASK_NAME="turn_switch"
TASK_CONFIG="demo_clean"
EPISODES=50
EVAL_STEP_LIMIT=400
STEP_LIMIT_CONFIG="${ROOT_DIR}/env_cfg/task_config/_eval_step_limit.yml"
RAW_DATA_DIR="${ROOT_DIR}/data/${TASK_CONFIG}/${TASK_NAME}/${EMBODIMENT}/data"
PROCESSED_DIR="${ACT_DIR}/processed_data/${TASK_CONFIG}/${TASK_NAME}/${EMBODIMENT}-${ACTION_TYPE}"
EVAL_DIR="${ROOT_DIR}/eval_result/${TASK_NAME}/ACT/${TASK_CONFIG}"

export PATH="${CONDA_BASE}/condabin:${CONDA_BASE}/bin:${PATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "===== turn_switch bs16 pipeline started $(date -Is) ====="
echo "root=${ROOT_DIR} env=${CONDA_ENV} gpu=${GPU_ID}"
echo "task=${TASK_NAME} config=${TASK_CONFIG} episodes=${EPISODES} eval_step_limit=${EVAL_STEP_LIMIT}"
echo "training: batch_size=16 grad_accum_steps=1 epochs=2000 lr=1e-5 kl_weight=10 chunk_size=50 hidden_dim=512 dim_feedforward=3200 seed=0"

python - "${GPU_ID}" <<'PY'
import sys
import torch

gpu_id = int(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
device = torch.cuda.get_device_properties(gpu_id)
if "RTX 3090" not in device.name or device.total_memory < 22 * 1024**3:
    raise SystemExit(
        f"Expected an RTX 3090 with 24 GB VRAM; got {device.name}, "
        f"{device.total_memory / 1024**3:.2f} GiB"
    )
print(f"GPU verified: {device.name}, {device.total_memory / 1024**3:.2f} GiB")
PY

verify_eval_step_limit() {
    local configured_limit
    configured_limit="$(awk -F: '$1 == "turn_switch" {gsub(/[[:space:]]/, "", $2); print $2}' "${STEP_LIMIT_CONFIG}")"
    if [[ "${configured_limit}" != "${EVAL_STEP_LIMIT}" ]]; then
        echo "Expected turn_switch eval step limit ${EVAL_STEP_LIMIT}; found ${configured_limit:-missing}" >&2
        return 1
    fi
}

verify_raw_data() {
    python - "${RAW_DATA_DIR}" "${EPISODES}" <<'PY'
import sys
from pathlib import Path

import h5py
import numpy as np

data_dir = Path(sys.argv[1])
expected = int(sys.argv[2])
files = sorted(data_dir.glob("episode_*.hdf5"))
if len(files) != expected:
    raise SystemExit(f"Expected {expected} raw HDF5 episodes; found {len(files)}")
contact_positive = 0
force_nonzero = 0
force_frames = 0
dimensions = set()
for path in files:
    with h5py.File(path, "r") as h5:
        observations = h5.get("observations")
        if observations is None or "force" not in observations or "contact_gt" not in observations:
            raise SystemExit(f"Missing /observations/force or /observations/contact_gt in {path}")
        force = np.asarray(observations["force"], dtype=np.float32)
        contact = np.asarray(observations["contact_gt"], dtype=np.float32)
        if force.ndim != 2 or force.shape[1] not in (6, 12) or not force.shape[0]:
            raise SystemExit(f"Unexpected force shape {force.shape} in {path}")
        if contact.ndim == 0 or contact.shape[0] != force.shape[0]:
            raise SystemExit(f"Force/contact frame mismatch in {path}")
        if not np.isfinite(force).all() or not np.isfinite(contact).all():
            raise SystemExit(f"Non-finite force/contact values in {path}")
        dimensions.add(force.shape[1])
        force_frames += force.shape[0]
        force_nonzero += int(np.count_nonzero(np.abs(force) > 1e-6))
        contact_positive += int((contact > 0.5).sum())
if contact_positive == 0:
    raise SystemExit("No positive contact_gt labels found across collected episodes")
if force_nonzero == 0:
    raise SystemExit("All collected force values are zero")
print(
    f"Raw HDF5 PASS: {len(files)} episodes, force dimensions={sorted(dimensions)}, "
    f"{force_frames} force frames, nonzero force values={force_nonzero}, "
    f"positive contact labels={contact_positive}"
)
PY
}

if [[ ! -d "${RAW_DATA_DIR}" ]] || [[ "$(find "${RAW_DATA_DIR}" -maxdepth 1 -type f -name 'episode_*.hdf5' | wc -l)" -lt "${EPISODES}" ]]; then
    echo "===== Collecting ${EPISODES} force demonstrations $(date -Is) ====="
    cd "${ROOT_DIR}"
    PYTHONUNBUFFERED=1 bash "${ROOT_DIR}/collect_data.sh" \
        "${TASK_NAME}" "${TASK_CONFIG}" "${GPU_ID}" "${EPISODES}"
fi

echo "===== Verifying raw force/contact data $(date -Is) ====="
verify_raw_data

echo "===== Converting HDF5 for ACT $(date -Is) ====="
cd "${ACT_DIR}"
bash "${ACT_DIR}/process_data.sh" \
    "${TASK_CONFIG}" "${TASK_NAME}" "${EMBODIMENT}" "${ACTION_TYPE}" \
    --episode_num "${EPISODES}"

echo "===== Verifying ACT force/contact data $(date -Is) ====="
python - "${PROCESSED_DIR}" "${EPISODES}" <<'PY'
import sys
from pathlib import Path

import h5py
import numpy as np

data_dir = Path(sys.argv[1])
expected = int(sys.argv[2])
files = sorted(data_dir.glob("episode_*.hdf5"))
if len(files) != expected:
    raise SystemExit(f"Expected {expected} processed HDF5 episodes; found {len(files)}")
contact_positive = 0
force_nonzero = 0
dimensions = set()
for path in files:
    with h5py.File(path, "r") as h5:
        observations = h5.get("observations")
        if observations is None or "force" not in observations or "contact_gt" not in observations:
            raise SystemExit(f"Missing converted /observations/force or /observations/contact_gt in {path}")
        force = np.asarray(observations["force"], dtype=np.float32)
        contact = np.asarray(observations["contact_gt"], dtype=np.float32)
        if force.ndim != 2 or force.shape[1] not in (6, 12) or not force.shape[0]:
            raise SystemExit(f"Unexpected converted force shape {force.shape} in {path}")
        if contact.ndim == 0 or contact.shape[0] != force.shape[0]:
            raise SystemExit(f"Converted force/contact frame mismatch in {path}")
        if not np.isfinite(force).all() or not np.isfinite(contact).all():
            raise SystemExit(f"Non-finite converted force/contact values in {path}")
        dimensions.add(force.shape[1])
        force_nonzero += int(np.count_nonzero(np.abs(force) > 1e-6))
        contact_positive += int((contact > 0.5).sum())
if contact_positive == 0:
    raise SystemExit("No positive converted contact_gt labels found")
if force_nonzero == 0:
    raise SystemExit("All converted force values are zero")
print(
    f"ACT HDF5 PASS: {len(files)} episodes, force dimensions={sorted(dimensions)}, "
    f"nonzero force values={force_nonzero}, positive contact labels={contact_positive}"
)
PY

verify_eval_step_limit
mkdir -p "${EVAL_DIR}"

train_and_eval() {
    local experiment=$1
    local gate=$2
    local disable_force=${3:-false}
    local checkpoint="${experiment}"

    echo "===== TRAIN ${experiment} gate=${gate} disable_force=${disable_force} $(date -Is) ====="
    local train_args=(
        "${TASK_NAME}" "${TASK_CONFIG}" "${EPISODES}" 0 "${GPU_ID}" "${checkpoint}"
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
        "${TASK_NAME}" "${TASK_NAME}" "${checkpoint}" "${EMBODIMENT}" "${ACTION_TYPE}" \
        0 "${GPU_ID}" "${GPU_ID}" "${CONDA_ENV}" "${CONDA_ENV}" \
        "${gate}" "${disable_force}" 50 "${TASK_CONFIG}" "${experiment}"
}

cd "${ACT_DIR}"
train_and_eval exp01_concat none false

echo "===== EVAL exp01_concat_visual_only reusing exp01_concat checkpoint $(date -Is) ====="
bash "${ACT_DIR}/eval.sh" \
    "${TASK_NAME}" "${TASK_NAME}" exp01_concat "${EMBODIMENT}" "${ACTION_TYPE}" \
    0 "${GPU_ID}" "${GPU_ID}" "${CONDA_ENV}" "${CONDA_ENV}" \
    none true 50 "${TASK_CONFIG}" exp01_concat_visual_only

train_and_eval exp02_hard hard false
train_and_eval exp03_attn cross_attn false
train_and_eval exp04_soft soft false

echo "===== SUMMARY $(date -Is) ====="
python "${ROOT_DIR}/summarize_turn_switch_metrics.py"
echo "===== turn_switch bs16 pipeline completed $(date -Is) ====="
