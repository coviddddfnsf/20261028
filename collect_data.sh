#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 TASK_NAME TASK_CONFIG GPU_ID [EPISODE_NUM]" >&2
    exit 2
fi

task_name=$1
task_config=$2
gpu_id=$3
episode_num=${4:-}

export CUDA_VISIBLE_DEVICES="${gpu_id}"
if [[ -z "${VK_ICD_FILENAMES:-}" && -f "${HOME}/.config/vulkan/icd.d/nvidia-egl.json" ]]; then
    export VK_ICD_FILENAMES="${HOME}/.config/vulkan/icd.d/nvidia-egl.json"
fi

python_args=()
if [[ -n "${episode_num}" ]]; then
    if [[ ! "${episode_num}" =~ ^[1-9][0-9]*$ ]]; then
        echo "EPISODE_NUM must be a positive integer." >&2
        exit 2
    fi
    python_args+=(--episode_num "${episode_num}")
fi
PYTHONWARNINGS=ignore::UserWarning \
PYTHONUNBUFFERED=1 python scripts/collect_data.py "${task_name}" "${task_config}" "${python_args[@]}"
# The third level is the embodiment directory (aloha_agilex by default),
# so clean caches for whichever embodiment was collected.
for cache_dir in "data/${task_config}/${task_name}"/*/.cache; do
    [[ -d "${cache_dir}" ]] || continue
    rm -rf -- "${cache_dir}"
done
