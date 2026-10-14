#!/bin/bash
# GSPO | Qwen3.5-35B-A3B Multimodal MoE | vLLM rollout | Megatron training | Ascend 910B | 4 nodes
set -xeuo pipefail

echo "***************************************** copy file *****************************************"

CACHE_ROOT=/cache/algorithm
PROJECT_ROOT=${CACHE_ROOT}
ROOT_PATH=/opt/huawei

rm -rf ${PROJECT_ROOT}

cp -r ${ROOT_PATH}/schedule-train/algorithm ${PROJECT_ROOT}
cd ${PROJECT_ROOT}

echo "PROJECT_ROOT=${PROJECT_ROOT}"
ls ${PROJECT_ROOT}

echo "***************************************** update cann *****************************************"

cp ${cq_qwen_datasets}/tools/cann_packages/cann852/Ascend-cann-toolkit_8.5.2_linux-aarch64.run ${PROJECT_ROOT}
cp ${cq_qwen_datasets}/tools/cann_packages/cann852/Ascend-cann-910b-ops_8.5.2_linux-aarch64.run ${PROJECT_ROOT}
cp ${cq_qwen_datasets}/tools/cann_packages/cann852/Ascend-cann-nnal_8.5.2_linux-aarch64.run ${PROJECT_ROOT}

chmod +777 ./Ascend-cann-*

./Ascend-cann-toolkit_8.5.2_linux-aarch64.run --install --quiet
set +u
source /home/ma-user/Ascend/cann-8.5.2/set_env.sh
set -u

./Ascend-cann-910b-ops_8.5.2_linux-aarch64.run --install --quiet
set +u
source /home/ma-user/Ascend/cann-8.5.2/set_env.sh
set -u

./Ascend-cann-nnal_8.5.2_linux-aarch64.run --install --quiet
set +u
source /home/ma-user/Ascend/nnal/atb/set_env.sh
set -u

echo "***************************************** set env *****************************************"

export CUDA_DEVICE_MAX_CONNECTIONS=1

export VLLM_USE_V1=1
export VLLM_USE_DEEPSEEK_V3=False
export VLLM_STATIC_GRAPH=False
export VLLM_USE_NEW_MODEL_IMPL=true
export VLLM_ENABLE_NZ_CACHE_FMT=False

export TASK_QUEUE_ENABLE=0
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1
unset PYTORCH_NPU_ALLOC_CONF

export HCCL_EXEC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export TORCH_DIST_TIMEOUT=3600
export ACL_DEVICE_SYNC_TIMEOUT=7200
export HCCL_IF_BASE_PORT=64000
export HCCL_OP_EXPANSION_MODE=AIV

export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RAY_USAGE_STATS_ENABLED=0
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

export NUMEXPR_MAX_THREADS=16
export TORCHINDUCTOR_COMPILE_THREADS=1

export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH

echo "***************************************** install dependencies *****************************************"

python3 -m pip install --upgrade pip
pip uninstall jsonnet -y || true
pip install --upgrade pip

cd ${PROJECT_ROOT}

pip install TransferQueue
pip install transferqueue-0.1.9.dev0-py3-none-any.whl

pip install triton_ascend==3.2.0

grep -v -E "triton-ascend|TransferQueue" requirements-npu.txt | pip install -r /dev/stdin
pip install pre-commit pylatexenc wandb qwen_vl_utils hf_transfer mathruler

pip install transformers==5.3.0 accelerate==1.2.0
pip install tokenizers==0.21.1 --no-deps
pip install vllm==0.18.0
pip install vllm-ascend==0.18.0

pip install -U hydra-core codetiming
pip install -U ray

pip install numpy av ftfy bs4 beartype pandas==2.0.3 diffusers==0.30.3 accelerate==0.32.1 pandarallel datasets qwen_vl_utils imageio imageio-ffmpeg jsonargparse peft==0.7.1 "pydantic>=2.10.4" gpytorch==1.14
pip install jsonnet==0.17.0
pip install torchdata rich tensordict
pip install flash-linear-attention==0.5.0
pip install datasets==4.3.0
pip install numpy==1.26.4
pip install triton-ascend==3.2.0
pip install -U hydra-core
pip install mathruler pylatexenc
pip install tokenizers==0.21.1

pip uninstall torch_c_dlpack_ext -y || true
pip install torch==2.9.0
pip install torch_npu==2.9.0

python -c "import torch; import torch_npu; print(torch_npu.npu.is_available())"

echo "***************************************** install Megatron-LM & MindSpeed & mbridge *****************************************"

# MindSpeed — bundled in schedule-train/algorithm, install to /cache/MindSpeed
if [ -d "${PROJECT_ROOT}/MindSpeed" ]; then
    rm -rf /cache/MindSpeed
    cp -r ${PROJECT_ROOT}/MindSpeed /cache/MindSpeed
    cd /cache/MindSpeed
    pip install -e .
    cd ${PROJECT_ROOT}
else
    echo "WARNING: MindSpeed not found in ${PROJECT_ROOT}, skipping"
fi

# Megatron-LM — check if already installed, otherwise clone from GitHub
if python -c "import megatron" 2>/dev/null; then
    echo "Megatron-LM already installed"
elif [ -d "${PROJECT_ROOT}/Megatron-LM" ]; then
    echo "Installing Megatron-LM from ${PROJECT_ROOT}/Megatron-LM..."
    pip install -e ${PROJECT_ROOT}/Megatron-LM
else
    echo "FATAL: Megatron-LM not installed and not found in algorithm/."
    echo "Download it first on a machine with internet access:"
    echo "  git clone --depth 1 --branch core_r0.16.0 https://github.com/NVIDIA/Megatron-LM.git"
    echo "Then bundle Megatron-LM/ into schedule-train/algorithm/ alongside MindSpeed."
    exit 1
fi

# mbridge
pip install mbridge

pip list | grep -iE "megatron|mindspeed|mbridge"

if [ -d "${PROJECT_ROOT}/transformers" ]; then
    cd ${PROJECT_ROOT}/transformers
    pip install -e .
    cd ${PROJECT_ROOT}
fi

pip list

echo "***************************************** user config *****************************************"

MODEL_PATH=${MODEL_PATH:-${cq_qwen_datasets}/opensource_multimodal_models_guiyang/Qwen/Qwen3.5-35B-A3B/Qwen/Qwen3___5-35B-A3B}

NNODES=${NNODES:-4}
NGPUS_PER_NODE=${MA_NUM_GPUS:-8}
NTRAINER_GPUS_PER_NODE=${NTRAINER_GPUS_PER_NODE:-8}

TRAIN_FILE=${TRAIN_FILE:-/opt/huawei/dataset/Geometry3K/geo3k/train.parquet}
VAL_FILE=${VAL_FILE:-/opt/huawei/dataset/Geometry3K/geo3k/test.parquet}

PROJECT_NAME=${PROJECT_NAME:-verl_gspo_qwen3_5_35b_a3b_geo3k}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_5_35b_a3b_gspo_megatron_910b_4n8c}

RUN_TAG=${RUN_TAG:-qwen3_5_35b-a3b-gspo-megatron-910b-4n8c-$(date +%Y%m%d_%H%M%S)}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-/opt/huawei/dataset/wyb_qwen_small/checkpoints/${RUN_TAG}}
LOG_PATH=${LOG_PATH:-/opt/huawei/dataset/wyb_qwen_small/logs/GSPO_EXP/${RUN_TAG}}
SAVELOG_FILE_NAME=${SAVELOG_FILE_NAME:-train.log}

mkdir -p ${DEFAULT_LOCAL_DIR}
mkdir -p ${LOG_PATH}

echo "***************************************** training hyperparams *****************************************"

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}

ACTOR_LR=${ACTOR_LR:-1e-6}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-3e-4}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-4e-4}

ACTOR_TP=${ACTOR_TP:-4}
ACTOR_PP=${ACTOR_PP:-2}
ACTOR_EP=${ACTOR_EP:-4}
ACTOR_ETP=${ACTOR_ETP:-1}
ACTOR_CP=${ACTOR_CP:-1}

ROLLOUT_TP=${ROLLOUT_TP:-8}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.99}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-10}

echo "***************************************** ray config *****************************************"

NODE_RANK=${VC_TASK_INDEX:-0}
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
MASTER_PORT=${MASTER_PORT:-6167}
DASHBOARD_PORT=${DASHBOARD_PORT:-8191}

export NODE_RANK MASTER_ADDR MASTER_PORT
export TMPDIR=/cache/algorithmzg_ray_tmp
mkdir -p ${TMPDIR}

export NPUS_PER_NODE=8
export WORLD_SIZE=$(( NNODES * NPUS_PER_NODE ))

readonly SOCKET_IFNAME="eth0"

export HCCL_SOCKET_IFNAME=$SOCKET_IFNAME
export GLOO_SOCKET_IFNAME=$SOCKET_IFNAME
export TP_SOCKET_IFNAME=$SOCKET_IFNAME
export ASCEND_GLOBAL_LOG_LEVEL=3

export RAY_gcs_server_rpc_server_thread_num=32
export RAY_gcs_server_request_timeout_seconds=1200
export RAY_timeout_ms=120000
export RAY_worker_register_timeout_seconds=1200

echo "Cleaning up old Ray processes..."
ray stop --force || true
sleep 5
rm -rf ${TMPDIR}/* || true
pkill -9 -f raylet || true
pkill -9 -f plasma_store || true
pkill -9 -f gcs_server || true
echo "Waiting 20s for NPU to release resources..."
sleep 20

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['${TRAIN_FILE}']"
    data.val_files="['${VAL_FILE}']"
    data.image_key=images
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=True
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_fused_kernels=False
)

ACTOR=(
    actor_rollout_ref.actor.strategy=megatron
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.calculate_entropy=False
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True

    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ACTOR_ETP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP}
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.grad_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True
    actor_rollout_ref.actor.megatron.use_remove_padding=False

    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}
    actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P}
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
)

REF=(
    actor_rollout_ref.ref.strategy=megatron
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}

    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${ACTOR_TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${ACTOR_PP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${ACTOR_EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ACTOR_ETP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${ACTOR_CP}
    actor_rollout_ref.ref.megatron.param_offload=True
    actor_rollout_ref.ref.megatron.use_mbridge=True
    actor_rollout_ref.ref.megatron.vanilla_mbridge=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NTRAINER_GPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.device=npu
    trainer.default_local_dir=${DEFAULT_LOCAL_DIR}
    trainer.val_before_train=True
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
)

EXTRA=(
    model_engine=megatron
)

echo "***************************************** start ray cluster *****************************************"

if [ "${NODE_RANK}" = "0" ]; then
    echo "Starting Ray HEAD node on ${MA_CURRENT_IP}..."

    ray start --head \
        --node-ip-address=${MA_CURRENT_IP} \
        --port=${MASTER_PORT} \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=${DASHBOARD_PORT} \
        --resources='{"NPU": '${NPUS_PER_NODE}'}'

    while true; do
        ray_status_output=$(ray status)
        npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
        npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
        device_count=$((npu_count_int / NPUS_PER_NODE))

        if [ "$device_count" -eq "$NNODES" ]; then
            echo "Ray cluster ready: ${device_count} nodes"
            ray status

            echo "***************************************** launch training *****************************************"

            python3 -m verl.trainer.main_ppo \
                "${DATA[@]}" \
                "${MODEL[@]}" \
                "${ACTOR[@]}" \
                "${ROLLOUT[@]}" \
                "${REF[@]}" \
                "${TRAINER[@]}" \
                "${EXTRA[@]}" \
                >> ${LOG_PATH}/${SAVELOG_FILE_NAME} 2>&1

            break
        fi

        echo "Waiting for Ray nodes. Current=${device_count}, expected=${NNODES}"
        sleep 10
    done
else
    while true; do
        sleep 5

        ray start \
            --node-ip-address=${MA_CURRENT_IP} \
            --address=${MASTER_ADDR}:${MASTER_PORT} \
            --resources='{"NPU": '${NPUS_PER_NODE}'}'

        if ray status; then
            echo "Worker connected to Ray cluster"
            break
        fi

        echo "Failed to connect Ray head, retrying..."
        sleep 10
    done
fi
