#!/usr/bin/env bash

# HPT Table-2 style evaluation | Qwen3.5-35B-A3B | Ascend 910B | MTP
#
# Required model inputs are HuggingFace-format directories:
#   SFT_MODEL_PATH=/path/to/sft_hf \
#   DFT_MODEL_PATH=/path/to/dft_hf \
#   GRPO_MODEL_PATH=/path/to/grpo_hf \
#   bash examples/hpt_eval/run_table2_mtp.sh
#
# HPT_MODEL_PATH is optional and adds the HPT row.  Use merge_checkpoints.sh
# first when a training output is still a sharded verl checkpoint.

set -euo pipefail
set -x

PROXY_ADDR="http://10.31.196.119:80"
export no_proxy="${no_proxy:+$no_proxy,}10.44.***.***"
export http_proxy="$PROXY_ADDR"
export https_proxy="$PROXY_ADDR"
export ftp_proxy="$PROXY_ADDR"
export HTTP_PROXY="$PROXY_ADDR"
export HTTP_PROXY="$PROXY_ADDR"
export FTP_PROXY="$PROXY_ADDR"

DATE_TAG=$(date +%Y%m%d_%H%M%S)
ROOT_PATH=${ROOT_PATH:-/opt/huawei}
PROJECT_ROOT=${PROJECT_ROOT:-/cache/algorithm}

# Match the MTP training job: every node receives the uploaded algorithm tree.
if [[ "$(pwd -P)" != "${PROJECT_ROOT}" ]]; then
    mkdir -p "${PROJECT_ROOT}"
    cp -r "${ROOT_PATH}/schedule-train/algorithm/." "${PROJECT_ROOT}/"
fi
cd "${PROJECT_ROOT}"

set +u
source /home/ma-user/Ascend/cann-8.5.2/set_env.sh
source /home/ma-user/Ascend/nnal/atb/set_env.sh
set -u

export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=1
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1
unset PYTORCH_NPU_ALLOC_CONF
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:${LD_LIBRARY_PATH:-}
export HCCL_EXEC_TIMEOUT=7200
export HCCL_IF_BASE_PORT=64000
export ACL_DEVICE_SYNC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export TORCH_DIST_TIMEOUT=3600
export HCCL_OP_EXPANSION_MODE=AIV
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export VLLM_RPC_TIMEOUT=600000

BASE_PATH=${zlt_data}
DATA_DIR=${DATA_DIR:-${BASE_PATH}/HPT_data}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_PATH}/runs/eval_result/HPT_table2_${DATE_TAG}}

# Fine-tuned checkpoints must be merged to HuggingFace format before rollout.
SFT_MODEL_PATH=${SFT_MODEL_PATH:-${BASE_PATH}/runs/sft/qwen35_35b_0922/qwen35b_sft_hf}
# DFT_MODEL_PATH=${DFT_MODEL_PATH:-${BASE_PATH}/runs/dft/qwen35_35b_0922/qwen35b_sft_hf}
# GRPO_MODEL_PATH=${GRPO_MODEL_PATH:-${BASE_PATH}/eval_models/qwen35b_grpo_hf}
HPT_MODEL_PATH=${HPT_MODEL_PATH:-}

# Defaults follow zlt_run_grpo_qwen35b_v3.sh and its commented validation set.
AIME24_FILE=${AIME24_FILE:-${DATA_DIR}/AIME24/test.parquet}
AIME25_FILE=${AIME25_FILE:-${DATA_DIR}/AIME25/test.parquet}
AMC_FILES=${AMC_FILES:-${DATA_DIR}/AMC23/test.parquet}
MATH500_FILE=${MATH500_FILE:-${DATA_DIR}/MATH-500/test.parquet}
MINERVA_FILE=${MINERVA_FILE:-${DATA_DIR}/Minerva/test.parquet}
OLYMPIAD_FILE=${OLYMPIAD_FILE:-${DATA_DIR}/Olympiad/test.parquet}
ARCC_FILE=${ARCC_FILE:-${DATA_DIR}/ARC-C/test.parquet}
GPQA_FILE=${GPQA_FILE:-${DATA_DIR}/GPQA/test.parquet}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-12888}
ROLLOUT_TP=${ROLLOUT_TP:-4}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.55}
EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-0.6}
EVAL_TOP_P=${EVAL_TOP_P:-0.95}
SCORER_WORKERS=${SCORER_WORKERS:-16}
RESUME_GENERATION=${RESUME_GENERATION:-1}
PYPI_INDEX_URL=${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}

NNODES=${NNODES:-${MA_NUM_HOSTS:-1}}
NPUS_PER_NODE=${NPUS_PER_NODE:-${MA_NUM_GPUS:-8}}
NODE_RANK=${VC_TASK_INDEX:-0}
MASTER_ADDR_SOURCE=${VC_WORKER_HOSTS:-${MA_CURRENT_IP:-127.0.0.1}}
MASTER_ADDR=${MASTER_ADDR_SOURCE%%,*}
MASTER_PORT=${MASTER_PORT:-6167}
DASHBOARD_PORT=${DASHBOARD_PORT:-8191}

export NODE_RANK MASTER_ADDR MASTER_PORT
export TMPDIR=${TMPDIR:-/cache/ray_tmp}
mkdir -p "${TMPDIR}" "${OUTPUT_ROOT}"

readonly SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}
export RAY_DEDUP_LOGS=0
export HCCL_SOCKET_IFNAME=${SOCKET_IFNAME}
export GLOO_SOCKET_IFNAME=${SOCKET_IFNAME}
export TP_SOCKET_IFNAME=${SOCKET_IFNAME}
export ASCEND_GLOBAL_LOG_LEVEL=3
export RAY_gcs_server_rpc_server_thread_num=32
export RAY_gcs_server_request_timeout_seconds=1200
export RAY_timeout_ms=120000
export RAY_worker_register_timeout_seconds=1200
export RAY_USAGE_STATS_ENABLED=0
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_HEARTBEAT_TIMEOUT_MS=60000
export RAY_gcs_actor_failure_grace_period_ms=60000

split_paths() {
    local value=$1
    value=${value//,/ }
    read -r -a SPLIT_RESULT <<< "${value}"
}

json_array() {
    local first=1
    local value
    printf '['
    for value in "$@"; do
        if (( first == 0 )); then printf ','; fi
        printf '"%s"' "${value}"
        first=0
    done
    printf ']'
}

require_files() {
    local missing=0
    local path
    for path in "$@"; do
        if [[ ! -f "${path}" ]]; then
            echo "Missing evaluation dataset: ${path}" >&2
            missing=1
        fi
    done
    if (( missing != 0 )); then
        echo "Set the corresponding *_FILE/AMC_FILES variables to the MTP parquet paths." >&2
        exit 2
    fi
}

generate_group() {
    local stage=$1
    local model_path=$2
    local sample_count=$3
    local output_path=$4
    shift 4
    local files=("$@")
    local files_json
    files_json=$(json_array "${files[@]}")

    if [[ "${RESUME_GENERATION}" == "1" && -s "${output_path}" ]]; then
        echo "Reuse existing rollout parquet: ${output_path}"
        return
    fi

    python3 -m verl.trainer.main_generation_server \
        trainer.nnodes="${NNODES}" \
        trainer.n_gpus_per_node="${NPUS_PER_NODE}" \
        data.train_files="${files_json}" \
        data.prompt_key=prompt \
        +data.output_path="${output_path}" \
        +data.progress_desc="Generating ${stage} avg@${sample_count}" \
        actor_rollout_ref.model.path="${model_path}" \
        actor_rollout_ref.model.trust_remote_code=True \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.temperature="${EVAL_TEMPERATURE}" \
        actor_rollout_ref.rollout.top_p="${EVAL_TOP_P}" \
        actor_rollout_ref.rollout.top_k=-1 \
        actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}" \
        actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
        actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
        actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
        actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.n="${sample_count}"
}

evaluate_stage() {
    local stage=$1
    local model_path=$2
    local stage_dir=${OUTPUT_ROOT}/${stage}
    local generated_dir=${stage_dir}/generated
    mkdir -p "${generated_dir}"

    if [[ ! -d "${model_path}" ]]; then
        echo "${stage}_MODEL_PATH is not a directory: ${model_path}" >&2
        echo "Merge the verl checkpoint first with examples/hpt_eval/merge_checkpoints.sh." >&2
        exit 2
    fi

    # Paper protocol: avg@32 for contest sets; avg@1 for the remaining sets.
    generate_group "${stage}" "${model_path}" 32 "${generated_dir}/avg32.parquet" "${AVG32_FILES[@]}"
    generate_group "${stage}" "${model_path}" 1 "${generated_dir}/avg1.parquet" "${AVG1_FILES[@]}"

    python3 -m verl.experimental.hpt.table2_eval score \
        --stage "${stage}" \
        --input "${generated_dir}/avg32.parquet" \
        --input "${generated_dir}/avg1.parquet" \
        --output-dir "${stage_dir}" \
        --workers "${SCORER_WORKERS}" \
        --max-response-tokens "${MAX_RESPONSE_LENGTH}"
}

split_paths "${AMC_FILES}"
AMC_PATHS=("${SPLIT_RESULT[@]}")
AVG32_FILES=("${AIME24_FILE}" "${AIME25_FILE}" "${AMC_PATHS[@]}")
AVG1_FILES=("${MATH500_FILE}" "${MINERVA_FILE}" "${OLYMPIAD_FILE}" "${ARCC_FILE}" "${GPQA_FILE}")
require_files "${AVG32_FILES[@]}" "${AVG1_FILES[@]}"

python3 -c 'import torch, torch_npu; assert torch_npu.npu.is_available(); print("NPU ready")'
if ! python3 -c 'import pandas, pyarrow, aiohttp, openai, sympy, latex2sympy2_extended, math_verify, pylatexenc, tqdm'; then
    python3 -m pip install \
        pandas pyarrow aiohttp openai sympy latex2sympy2_extended math-verify pylatexenc tqdm \
        -i "${PYPI_INDEX_URL}"
fi
python3 -c 'import pandas, pyarrow, aiohttp, openai, sympy, latex2sympy2_extended, math_verify, pylatexenc, tqdm; print("evaluation dependencies ready")'

if [[ "${NODE_RANK}" == "0" ]]; then
    ray start --head \
        --node-ip-address="${MA_CURRENT_IP}" \
        --port="${MASTER_PORT}" \
        --dashboard-host=0.0.0.0 \
        --dashboard-port="${DASHBOARD_PORT}" \
        --resources="{\"NPU\": ${NPUS_PER_NODE}}"

    while true; do
        ray_status_output=$(ray status)
        npu_count=$(grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' <<< "${ray_status_output}" | head -n 1 || true)
        npu_count_int=$(awk '{print int($1)}' <<< "${npu_count:-0}")
        device_count=$((npu_count_int / NPUS_PER_NODE))
        if [[ "${device_count}" -eq "${NNODES}" ]]; then
            break
        fi
        echo "Waiting for Ray nodes. Current=${device_count}, expected=${NNODES}"
        sleep 10
    done
    ray status
    export RAY_ADDRESS="${MA_CURRENT_IP}:${MASTER_PORT}"

    STAGES=(SFT) # DFT) # GRPO)
    MODEL_PATHS=("${SFT_MODEL_PATH}") # "${DFT_MODEL_PATH}") # "${GRPO_MODEL_PATH}")
    if [[ -n "${HPT_MODEL_PATH}" ]]; then
        STAGES+=(HPT)
        MODEL_PATHS+=("${HPT_MODEL_PATH}")
    fi

    RUN_DIR_ARGS=()
    for index in "${!STAGES[@]}"; do
        evaluate_stage "${STAGES[index]}" "${MODEL_PATHS[index]}" 2>&1 | tee -a "${OUTPUT_ROOT}/${STAGES[index]}.log"
        RUN_DIR_ARGS+=(--run-dir "${OUTPUT_ROOT}/${STAGES[index]}")
    done

    python3 -m verl.experimental.hpt.table2_eval aggregate \
        "${RUN_DIR_ARGS[@]}" \
        --output-dir "${OUTPUT_ROOT}/report"
    echo "HPT Table-2 evaluation finished: ${OUTPUT_ROOT}/report/table2.md"
else
    ray start \
        --node-ip-address="${MA_CURRENT_IP}" \
        --address="${MASTER_ADDR}:${MASTER_PORT}" \
        --resources="{\"NPU\": ${NPUS_PER_NODE}}"
    sleep infinity
fi
