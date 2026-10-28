#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACT_DIR="${ROOT_DIR}/XPolicyLab/policy/ACT"
CONDA_BASE="${CONDA_BASE:-/root/miniconda3}"
CONDA_ENV="${CONDA_ENV:-base}"
GPU_ID="${GPU_ID:-0}"
EMBODIMENT="aloha_agilex"
ACTION_TYPE="joint"
TASK_NAME="handover_block"
TASK_CONFIG="demo_clean"
EPISODES=50
EVAL_STEP_LIMIT=800
STEP_LIMIT_CONFIG="${ROOT_DIR}/env_cfg/task_config/_eval_step_limit.yml"

export PATH="${CONDA_BASE}/condabin:${CONDA_BASE}/bin:${PATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

RAW_DIR="${ROOT_DIR}/data/${TASK_CONFIG}/${TASK_NAME}"
PROCESSED_DIR="${ACT_DIR}/processed_data/${TASK_CONFIG}/${TASK_NAME}"
EVAL_DIR="${ROOT_DIR}/eval_result/${TASK_NAME}"
CHECKPOINT_DIR="${ACT_DIR}/checkpoints"

echo "===== Handover Block bs16 pipeline started $(date -Is) ====="
echo "root=${ROOT_DIR} env=${CONDA_ENV} gpu=${GPU_ID}"

verify_eval_step_limit() {
    local configured_limit
    configured_limit="$(awk -F: '$1 == "handover_block" {gsub(/[[:space:]]/, "", $2); print $2}' "${STEP_LIMIT_CONFIG}")"
    if [[ "${configured_limit}" != "${EVAL_STEP_LIMIT}" ]]; then
        echo "Expected handover_block eval step limit ${EVAL_STEP_LIMIT}; found ${configured_limit:-missing}" >&2
        return 1
    fi
}

echo "===== Cleaning Handover Block caches $(date -Is) ====="
RAW_DATA_DIR="${RAW_DIR}/${EMBODIMENT}/data"
if python - "${RAW_DATA_DIR}" "${EPISODES}" <<'PY'
import sys
from pathlib import Path

import h5py
import numpy as np

data_dir = Path(sys.argv[1])
expected = int(sys.argv[2])
files = sorted(data_dir.glob("episode_*.hdf5"))
if len(files) != expected:
    raise SystemExit(1)
for path in files:
    try:
        with h5py.File(path, "r") as h5:
            observations = h5["observations"]
            force = observations["force"]
            contact = observations["contact_gt"]
            if force.ndim != 2 or force.shape[1] not in (6, 12):
                raise SystemExit(1)
            if not np.isfinite(force[()]).all() or not np.isfinite(contact[()]).all():
                raise SystemExit(1)
    except (OSError, KeyError, ValueError):
        raise SystemExit(1)
PY
then
    echo "Keeping complete ${EPISODES}-episode force dataset at ${RAW_DATA_DIR}"
else
    echo "No complete force dataset found; clearing raw collection output"
    rm -rf -- "${RAW_DIR}"
fi
rm -rf -- "${EVAL_DIR}" "${PROCESSED_DIR}"
for experiment in exp01_concat exp02_hard exp03_attn exp04_soft; do
    rm -rf -- "${CHECKPOINT_DIR}/handover_block_${experiment}"
done

if [[ ! -d "${RAW_DATA_DIR}" ]]; then
    echo "===== Collecting ${EPISODES} force demonstrations $(date -Is) ====="
    cd "${ROOT_DIR}"
    PYTHONUNBUFFERED=1 bash "${ROOT_DIR}/collect_data.sh" "${TASK_NAME}" "${TASK_CONFIG}" "${GPU_ID}"
else
    echo "===== Reusing complete ${EPISODES}-episode force demonstrations $(date -Is) ====="
fi

echo "===== Verifying raw force/contact HDF5 data $(date -Is) ====="
python - "${RAW_DIR}/aloha_agilex/data" "${EPISODES}" <<'PY'
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
force_frames = 0
dimensions = set()
for path in files:
    with h5py.File(path, "r") as h5:
        observations = h5.get("observations")
        if observations is None or "force" not in observations:
            raise SystemExit(f"Missing /observations/force in {path}")
        if "contact_gt" not in observations:
            raise SystemExit(f"Missing /observations/contact_gt in {path}")
        force = np.asarray(observations["force"], dtype=np.float32)
        contact = np.asarray(observations["contact_gt"], dtype=np.float32)
        if force.ndim != 2 or force.shape[1] not in (6, 12) or not force.shape[0]:
            raise SystemExit(f"Unexpected force shape {force.shape} in {path}")
        if not np.isfinite(force).all() or not np.isfinite(contact).all():
            raise SystemExit(f"Non-finite force/contact values in {path}")
        dimensions.add(force.shape[1])
        force_frames += force.shape[0]
        contact_positive += int((contact > 0.5).sum())
if contact_positive == 0:
    raise SystemExit("No positive contact_gt labels found across collected episodes")
print(
    f"Raw HDF5 PASS: {len(files)} episodes, force dimensions={sorted(dimensions)}, "
    f"{force_frames} force frames, positive contact labels={contact_positive}"
)
PY

echo "===== Converting HDF5 for ACT $(date -Is) ====="
cd "${ACT_DIR}"
bash "${ACT_DIR}/process_data.sh" \
    "${TASK_CONFIG}" "${TASK_NAME}" "${EMBODIMENT}" "${ACTION_TYPE}" \
    --episode_num "${EPISODES}"

echo "===== Verifying ACT force/contact data $(date -Is) ====="
python - "${PROCESSED_DIR}/${EMBODIMENT}-${ACTION_TYPE}" "${EPISODES}" <<'PY'
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
dimensions = set()
for path in files:
    with h5py.File(path, "r") as h5:
        observations = h5.get("observations")
        if observations is None or "force" not in observations:
            raise SystemExit(f"Missing converted /observations/force in {path}")
        if "contact_gt" not in observations:
            raise SystemExit(f"Missing converted /observations/contact_gt in {path}")
        force = np.asarray(observations["force"], dtype=np.float32)
        contact = np.asarray(observations["contact_gt"], dtype=np.float32)
        if force.ndim != 2 or force.shape[1] not in (6, 12) or not force.shape[0]:
            raise SystemExit(f"Unexpected converted force shape {force.shape} in {path}")
        if not np.isfinite(force).all() or not np.isfinite(contact).all():
            raise SystemExit(f"Non-finite converted force/contact values in {path}")
        dimensions.add(force.shape[1])
        contact_positive += int((contact > 0.5).sum())
if contact_positive == 0:
    raise SystemExit("No positive converted contact_gt labels found")
print(
    f"ACT HDF5 PASS: {len(files)} episodes, force dimensions={sorted(dimensions)}, "
    f"positive contact labels={contact_positive}"
)
PY

train_and_eval() {
    local experiment=$1
    local gate=$2
    local checkpoint="handover_block_${experiment}"

    echo "===== TRAIN ${experiment} gate=${gate} $(date -Is) ====="
    bash "${ACT_DIR}/train.sh" \
        "${TASK_NAME}" "${TASK_CONFIG}" "${EPISODES}" 0 "${GPU_ID}" "${checkpoint}" \
        --gate_type "${gate}" \
        --batch_size 16 \
        --num_epochs 2000

    echo "===== EVAL ${experiment} gate=${gate} $(date -Is) ====="
    verify_eval_step_limit
    bash "${ACT_DIR}/eval.sh" \
        "${TASK_NAME}" "${TASK_NAME}" "${checkpoint}" "${EMBODIMENT}" "${ACTION_TYPE}" \
        0 "${GPU_ID}" "${GPU_ID}" "${CONDA_ENV}" "${CONDA_ENV}" \
        "${gate}" false 50 "${TASK_CONFIG}" "${experiment}"
}

cd "${ACT_DIR}"
train_and_eval exp01_concat none

echo "===== EVAL exp01_concat_visual_only reusing exp01_concat checkpoint $(date -Is) ====="
verify_eval_step_limit
bash "${ACT_DIR}/eval.sh" \
    "${TASK_NAME}" "${TASK_NAME}" handover_block_exp01_concat "${EMBODIMENT}" "${ACTION_TYPE}" \
    0 "${GPU_ID}" "${GPU_ID}" "${CONDA_ENV}" "${CONDA_ENV}" \
    none true 50 "${TASK_CONFIG}" exp01_concat_visual_only

train_and_eval exp02_hard hard
train_and_eval exp03_attn cross_attn
train_and_eval exp04_soft soft

echo "===== SUMMARY $(date -Is) ====="
python "${ROOT_DIR}/summarize_handover_metrics.py"
echo "===== Handover Block pipeline completed $(date -Is) ====="
