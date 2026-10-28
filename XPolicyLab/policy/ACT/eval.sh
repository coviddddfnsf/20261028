#!/bin/bash
set -euo pipefail
# ==================== Argument definitions ====================
policy_name=ACT
bench_name=${1}
task_name=${2}
ckpt_name=${3}  # task_name
env_cfg_type=${4}
action_type=${5}
seed=${6}
policy_gpu_id=${7}
env_gpu_id=${8}
policy_conda_env=${9}
eval_env_conda_env=${10}
gate_type=${11:-none}
disable_force=${12:-false}
test_num=${13:-50}
task_config=${14:-demo_clean}
result_name=${15:-${ckpt_name}}
max_seed_attempts=${16:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # Current Dir
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/setup_eval_env_client.sh"

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip="localhost"

additional_info="ckpt_name=${ckpt_name},ckpt_setting=${result_name},action_type=${action_type},gate_type=${gate_type},disable_force=${disable_force}"
if [[ -n "${max_seed_attempts}" ]]; then
    if [[ ! "${max_seed_attempts}" =~ ^[1-9][0-9]*$ ]]; then
        echo "max_seed_attempts must be a positive integer." >&2
        exit 2
    fi
    eval_args_file=$(mktemp)
    printf '%s\n%s\n' '--max_seed_attempts' "${max_seed_attempts}" > "${eval_args_file}"
    export ROBOTWIN_EVAL_ARGS_FILE="${eval_args_file}"
fi

# Define cleanup to terminate background processes when the script exits
cleanup(){
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo -e "\033[31m[CLEANUP] Killing server PID=${SERVER_PID}\033[0m"
        kill "${SERVER_PID}" 2>/dev/null || true
    fi
    if [[ -n "${eval_args_file:-}" ]]; then
        rm -f "${eval_args_file}"
    fi
}
trap cleanup EXIT

echo "[MAIN] start server, policy_server_port=${policy_server_port}"

bash "${SERVER_SCRIPT}" \
    "${bench_name}" \
    "${task_name}" \
    "${ckpt_name}" \
    "${env_cfg_type}" \
    "${action_type}" \
    "${seed}" \
    "${policy_gpu_id}" \
    "${policy_conda_env}" \
    "${policy_server_port}" \
    localhost \
    "${gate_type}" \
    "${disable_force}" &

SERVER_PID=$!

bash "${UTILS_DIR}/wait_for_policy_server.sh" "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" "Policy server" 1200

echo "[MAIN] start client, server=${policy_server_ip}:${policy_server_port}"

bash "${CLIENT_SCRIPT}" \
    "${bench_name}" \
    "${task_name}" \
    "${ckpt_name}" \
    "${env_cfg_type}" \
    "${action_type}" \
    "${seed}" \
    "${env_gpu_id}" \
    "${eval_env_conda_env}" \
    "${additional_info}" \
    "${policy_server_port}" \
    "${policy_server_ip}" \
    ws \
    "${task_config}" \
    "${test_num}"

echo "[MAIN] eval finished"