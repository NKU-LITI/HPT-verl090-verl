#!/usr/bin/env bash

# Convert the SFT/DFT/GRPO sharded checkpoints produced by this repository to
# HuggingFace directories accepted by run_table2_mtp.sh.

set -euo pipefail

BASE_PATH=${BASE_PATH:-${zlt_data:?zlt_data is not set}}
SFT_CKPT=${SFT_CKPT:?set SFT_CKPT to an SFT global_step_* directory}
DFT_CKPT=${DFT_CKPT:?set DFT_CKPT to a DFT global_step_* directory}
GRPO_CKPT=${GRPO_CKPT:?set GRPO_CKPT to a GRPO global_step_* directory or its actor directory}
SFT_BACKEND=${SFT_BACKEND:-fsdp}
DFT_BACKEND=${DFT_BACKEND:-fsdp}
GRPO_BACKEND=${GRPO_BACKEND:-fsdp}
MERGED_ROOT=${MERGED_ROOT:-${BASE_PATH}/eval_models}

SFT_MODEL_PATH=${SFT_MODEL_PATH:-${MERGED_ROOT}/qwen35b_sft_hf}
DFT_MODEL_PATH=${DFT_MODEL_PATH:-${MERGED_ROOT}/qwen35b_dft_hf}
GRPO_MODEL_PATH=${GRPO_MODEL_PATH:-${MERGED_ROOT}/qwen35b_grpo_hf}

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

mkdir -p "${MERGED_ROOT}"
merge_one SFT "${SFT_BACKEND}" "${SFT_CKPT}" "${SFT_MODEL_PATH}"
merge_one DFT "${DFT_BACKEND}" "${DFT_CKPT}" "${DFT_MODEL_PATH}"
GRPO_ACTOR_CKPT=$(resolve_grpo_actor "${GRPO_CKPT}")
merge_one GRPO "${GRPO_BACKEND}" "${GRPO_ACTOR_CKPT}" "${GRPO_MODEL_PATH}"

printf 'Merged model paths:\nSFT_MODEL_PATH=%s\nDFT_MODEL_PATH=%s\nGRPO_MODEL_PATH=%s\n' \
    "${SFT_MODEL_PATH}" "${DFT_MODEL_PATH}" "${GRPO_MODEL_PATH}"
