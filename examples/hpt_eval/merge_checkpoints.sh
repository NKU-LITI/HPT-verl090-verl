#!/bin/bash
set -ex

PROXY_ADDR="http://10.31.196.119:80"
export no_proxy="${no_proxy:+$no_proxy,}10.44.***.***"
export http_proxy="$PROXY_ADDR"
export https_proxy="$PROXY_ADDR"
export ftp_proxy="$PROXY_ADDR"
export HTTP_PROXY="$PROXY_ADDR"
export HTTP_PROXY="$PROXY_ADDR"
export FTP_PROXY="$PROXY_ADDR"


# pip uninstall transformers -y
# # pip install transformers
# pip install "transformers>=4.51.0"
# # cd ${PROJECT_ROOT}/transformers
# pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers==5.10.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --upgrade accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install aiohttp aiofiles math-verify -i https://pypi.tuna.tsinghua.edu.cn/simple

BASE_PATH=${zlt_data}
SFT_CKPT=${SFT_CKPT:-${zlt_data}/runs/dft/qwen35_35b_1015/checkpoint/global_step_550}
DFT_CKPT=${DFT_CKPT:-${zlt_data}/runs/dft/qwen35_35b_1015/checkpoint/global_step_440}
# GRPO_CKPT=${GRPO_CKPT:?set GRPO_CKPT to a GRPO global_step_* directory or its actor directory}
SFT_BACKEND=${SFT_BACKEND:-fsdp}
DFT_BACKEND=${DFT_BACKEND:-fsdp}
# GRPO_BACKEND=${GRPO_BACKEND:-fsdp}
# MERGED_ROOT=${MERGED_ROOT:-${BASE_PATH}/runs/eval_models}

SFT_MODEL_PATH=${SFT_MODEL_PATH:-${BASE_PATH}/runs/dft/qwen35_35b_1015/checkpoint/global_step_550_hf}
DFT_MODEL_PATH=${DFT_MODEL_PATH:-${BASE_PATH}/runs/dft/qwen35_35b_1015/checkpoint/global_step_440_hf}
# GRPO_MODEL_PATH=${GRPO_MODEL_PATH:-${MERGED_ROOT}/qwen35b_grpo_hf}

resolve_grpo_actor() {
    local path=$1
    if [[ -d "${path}/actor" ]]; then
        printf '%s' "${path}/actor"
    else
        printf '%s' "${path}"
    fi
}

merge_one() {
    local label=$1
    local backend=$2
    local checkpoint=$3
    local target=$4
    if [[ ! -d "${checkpoint}" ]]; then
        echo "${label} checkpoint does not exist: ${checkpoint}" >&2
        exit 2
    fi
    if [[ -f "${target}/config.json" ]]; then
        echo "${label}: reuse existing merged model ${target}"
        return
    fi
    mkdir -p "${target}"
    python3 -m verl.model_merger merge \
        --backend "${backend}" \
        --local_dir "${checkpoint}" \
        --target_dir "${target}" \
        --trust-remote-code
}

# mkdir -p "${MERGED_ROOT}"
merge_one SFT "${SFT_BACKEND}" "${SFT_CKPT}" "${SFT_MODEL_PATH}"
merge_one DFT "${DFT_BACKEND}" "${DFT_CKPT}" "${DFT_MODEL_PATH}"
# GRPO_ACTOR_CKPT=$(resolve_grpo_actor "${GRPO_CKPT}")
# merge_one GRPO "${GRPO_BACKEND}" "${GRPO_ACTOR_CKPT}" "${GRPO_MODEL_PATH}"

printf 'Merged model paths:\nSFT_MODEL_PATH=%s\nDFT_MODEL_PATH=%s\nGRPO_MODEL_PATH=%s\n' \
    "${SFT_MODEL_PATH}" "${DFT_MODEL_PATH}" # "${GRPO_MODEL_PATH}"
