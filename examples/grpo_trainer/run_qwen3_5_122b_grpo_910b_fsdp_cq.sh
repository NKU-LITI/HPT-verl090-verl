#!/bin/bash

# GRPO | Qwen3.5-4B | Geo3K | vLLM rollout | FSDP | Ascend 910B multi-node
# 多节点版本：纯 GRPO，无 Teacher/Distillation
#
# dependency: NPU vllm==0.18.0, vllm-ascend@<54879467>, transformers@<cc7ab9be>

echo "***************************************** copy file *****************************************"
WORKDIR=`pwd`
echo "inital dictionary: $WORKDIR"
PROJECT_ROOT=/cache/algorithm

ROOT_PATH="/opt/huawei"
cp -r ${ROOT_PATH}/schedule-train/algorithm ${PROJECT_ROOT}
ls ${PROJECT_ROOT}

cp ${cq_qwen_datasets}/tools/cann_packages/cann852/Ascend-cann-toolkit_8.5.2_linux-aarch64.run ${PROJECT_ROOT}
cp ${cq_qwen_datasets}/tools/cann_packages/cann852/Ascend-cann-910b-ops_8.5.2_linux-aarch64.run ${PROJECT_ROOT}
cp ${cq_qwen_datasets}/tools/cann_packages/cann852/Ascend-cann-nnal_8.5.2_linux-aarch64.run ${PROJECT_ROOT}

ls ${PROJECT_ROOT}
echo "***************************************** update cann *****************************************"
cd ${PROJECT_ROOT}
chmod +777 ./Ascend-cann-*
./Ascend-cann-toolkit_8.5.2_linux-aarch64.run --install --quiet
source /home/ma-user/Ascend/cann-8.5.2/set_env.sh
./Ascend-cann-910b-ops_8.5.2_linux-aarch64.run --install --quiet
source /home/ma-user/Ascend/cann-8.5.2/set_env.sh
./Ascend-cann-nnal_8.5.2_linux-aarch64.run --install --quiet
source /home/ma-user/Ascend/nnal/atb/set_env.sh

echo "--------------------- adapt MTP end!-----------------------"
export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=1
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1
unset PYTORCH_NPU_ALLOC_CONF

export ACL_OP_COMPILER_CACHE_DIR=$ACL_OP_COMPILER_CACHE_DIR
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_EXEC_TIMEOUT=7200
export HCCL_IF_BASE_PORT=64000
export ACL_DEVICE_SYNC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export TORCH_DIST_TIMEOUT=3600
export HCCL_OP_EXPANSION_MODE=AIV

echo "***************************************** Installing Pytorch *****************************************"
python3 -m pip install --upgrade pip
pip uninstall jsonnet -y || true
pip install --upgrade pip

cd ${PROJECT_ROOT}
pip install TransferQueue
pip install transferqueue-0.1.9.dev0-py3-none-any.whl
pip install torch==2.9.0
pip install torch_npu==2.9.0
pip uninstall torch_c_dlpack_ext -y || true
pip install triton_ascend==3.2.0
pip install torch==2.9.0
pip install torch_npu==2.9.0
python -c "import torch;import torch_npu;print(torch_npu.npu.is_available())"


echo "*****************************************Start install transformers/vllm/vllm-ascend*****************************************"
cd ${PROJECT_ROOT}
pip install -r requirements-npu.txt
pip install pre-commit pylatexenc wandb qwen_vl_utils hf_transfer mathruler

pip install tokenizers==0.21.1 transformers==5.3.0 accelerate==1.2.0

pip install vllm==0.18.0
pip install vllm-ascend==0.18.0

pip install -U hydra-core codetiming

# ---- user-adjustable ----
# MODEL_SIZE: 4B | 35B
MODEL_SIZE=${MODEL_SIZE:-4B}

# /opt/huawei/dataset/cq_qwen_datasets/opensource_multimodal_models_guiyang/Qwen/Qwen3.5-397B-A17B/Qwen/Qwen3___5-397B-A17B
# dataset/cq_qwen_datasets/opensource_multimodal_models_guiyang/Qwen/Qwen3.5-122B-A10B
# 模型路径 & 多节点配置（按模型规模预设）
if [ "${MODEL_SIZE}" = "35B" ]; then
    MODEL_PATH=${MODEL_PATH:-${cq_qwen_datasets}/opensource_multimodal_models_guiyang/Qwen/Qwen3.5-122B-A10B}
    NNODES=${NNODES:-$MA_NUM_HOSTS}
    EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_5_35b_geo3k_grpo_910b_multi}
else
    MODEL_PATH=${MODEL_PATH:-/opt/huawei/dataset/wyb_qwen_small/Qwen3.5-4B}
    NNODES=${NNODES:-2}
    EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_5_4b_geo3k_grpo_910b_multi}
fi

NGPUS_PER_NODE=${MA_NUM_GPUS:-8}

# GRPO 不需要区分 Student/Teacher，整节点资源全给 actor/rollout/ref
NTRAINER_GPUS_PER_NODE=${NTRAINER_GPUS_PER_NODE:-8}

# /opt/huawei/dataset/cq_qwen_datasets/exp_train_json/train_rl_datasets_hard_8437_middle_1563_with_q_0724.parquet
# /opt/huawei/dataset/cq_qwen_datasets/exp_train_json/test_rl_datasets_hard_8437_middle_1563_math_0724.parquet
# /opt/huawei/dataset/Geometry3K/geo3k/train.parquet
# /opt/huawei/dataset/Geometry3K/geo3k/test.parquet
TRAIN_FILE=${TRAIN_FILE:-/opt/huawei/dataset/cq_qwen_datasets/exp_train_json/train_rl_datasets_hard_8437_middle_1563_with_q_0724.parquet}
TEST_FILE=${TEST_FILE:-/opt/huawei/dataset/cq_qwen_datasets/exp_train_json/test_rl_datasets_hard_8437_middle_1563_math_0724.parquet}

# verl_grpo_geo3k
PROJECT_NAME=${PROJECT_NAME:-verl_grpo_math1w}

# TRAIN_BATCH_SIZE 和 PPO_MINI_BATCH_SIZE设置成卡数
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}

ACTOR_LR=${ACTOR_LR:-1e-6}
PPO_EPOCHS=${PPO_EPOCHS:-1}
CLIP_RATIO=${CLIP_RATIO:-0.2}
GRAD_CLIP=${GRAD_CLIP:-1.0}

ROLLOUT_TP=${ROLLOUT_TP:-8}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.75}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.99}

# ${cq_qwen_datasets}/saved_models/rl_ckpt
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-${cq_qwen_datasets}/saved_models/rl_ckpt/${EXP_NAME}}
# ${cq_qwen_datasets}/xunlianlog/RL_EXP/${EXP_NAME}
LOG_PATH=${LOG_PATH:-${cq_qwen_datasets}/xunlianlog/RL_EXP}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
SAVE_FREQ=${SAVE_FREQ:-10}
VAL_FREQ=${VAL_FREQ:-5}

MAX_MODEL_LEN=${MAX_MODEL_LEN:-3200}

### 启动分布式网络配置（多节点）
NODE_RANK=${VC_TASK_INDEX:-0}
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
MASTER_PORT=${MASTER_PORT:-6167}
DASHBOARD_PORT=${DASHBOARD_PORT:-8191}

export NODE_RANK MASTER_ADDR MASTER_PORT
export TMPDIR=/cache/ray_tmp
mkdir -p /cache/ray_tmp

export NPUS_PER_NODE=8
export WORLD_SIZE=$(( NNODES * NPUS_PER_NODE ))

readonly SOCKET_IFNAME="eth0"
# # 确保使用正确的网卡名称
# export GLOO_SOCKET_IFNAME=eth0
# export TP_SOCKET_IFNAME=eth0
# # 尝试强制 vLLM 使用特定的本地地址进行初始化
# export VLLM_RPC_TORCH_DIST_BACKEND=gloo

pip install -U "ray"

pip install numpy av ftfy bs4 beartype pandas==2.0.3 diffusers==0.30.3 accelerate==0.32.1 pandarallel datasets qwen_vl_utils imageio imageio-ffmpeg jsonargparse peft==0.7.1 "pydantic>=2.10.4" gpytorch==1.14

pip install jsonnet==0.17.0
pip install torchdata rich
pip install tensordict

# ================= Ray 资源清理与环境就绪 =================
echo "Cleaning up old Ray processes..."
ray stop --force || true
sleep 5
rm -rf /cache/ray_tmp/* || true
pkill -9 -f raylet || true
pkill -9 -f plasma_store || true
pkill -9 -f gcs_server || true
echo "Waiting 20s for NPU to release resources..."
sleep 20

# 0.5.0 0.5.2 
pip install flash-linear-attention==0.5.2

cd ${PROJECT_ROOT}
pip install "datasets==4.3.0"
pip install numpy==1.26.4
pip install triton-ascend==3.2.0
pip install -U hydra-core

pip uninstall torch_c_dlpack_ext -y || true
pip install torch==2.9.0
pip install torch_npu==2.9.0

pip list

export RAY_DEDUP_LOGS=0
export HCCL_SOCKET_IFNAME=$SOCKET_IFNAME
export GLOO_SOCKET_IFNAME=$SOCKET_IFNAME
export TP_SOCKET_IFNAME=$SOCKET_IFNAME
export ASCEND_GLOBAL_LOG_LEVEL=3
export HYDRA_FULL_ERROR=1
export RAY_gcs_server_rpc_server_thread_num=32
export RAY_gcs_server_request_timeout_seconds=1200
export RAY_timeout_ms=120000
export RAY_worker_register_timeout_seconds=1200
export RAY_USAGE_STATS_ENABLED=0
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export VLLM_RPC_TIMEOUT=600000

# transformers==5.3.0
pip install mathruler pylatexenc

# 安装指定版本的transformer
pip install tokenizers==0.21.1
cd ${PROJECT_ROOT}/transformers
pip install -e .
cd ${PROJECT_ROOT}
pip list

export RAY_HEARTBEAT_TIMEOUT_MS=60000
export RAY_gcs_actor_failure_grace_period_ms=60000

# ================= 多节点：启动 Ray 集群 =================
if [ "${NODE_RANK}" = "0" ]; then
    # ---- 头节点：启动 Ray Head ----
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

            mkdir -p ${LOG_PATH}

            python3 -m verl.trainer.main_ppo \
                algorithm.adv_estimator=grpo \
                algorithm.use_kl_in_reward=False \
                data.train_files="['${TRAIN_FILE}']" \
                data.val_files="['${TEST_FILE}']" \
                data.image_key=images \
                data.train_batch_size=${TRAIN_BATCH_SIZE} \
                data.max_prompt_length=${MAX_PROMPT_LENGTH} \
                data.max_response_length=${MAX_RESPONSE_LENGTH} \
                data.filter_overlong_prompts=True \
                data.truncation='error' \
                data.shuffle=True \
                actor_rollout_ref.model.path="${MODEL_PATH}" \
                actor_rollout_ref.model.use_remove_padding=True \
                actor_rollout_ref.model.enable_gradient_checkpointing=True \
                actor_rollout_ref.actor.strategy=fsdp2 \
                actor_rollout_ref.actor.use_torch_compile=False \
                actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
                actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO} \
                actor_rollout_ref.actor.grad_clip=${GRAD_CLIP} \
                actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS} \
                actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
                actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
                actor_rollout_ref.actor.use_dynamic_bsz=False \
                actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
                actor_rollout_ref.actor.use_kl_loss=True \
                actor_rollout_ref.actor.kl_loss_coef=0.01 \
                actor_rollout_ref.actor.kl_loss_type=low_var_kl \
                actor_rollout_ref.actor.entropy_coeff=0 \
                actor_rollout_ref.actor.calculate_entropy=False \
                actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
                actor_rollout_ref.actor.fsdp_config.param_offload=True \
                actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
                actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
                actor_rollout_ref.actor.checkpoint.save_contents="['model']" \
                actor_rollout_ref.ref.strategy=fsdp2 \
                actor_rollout_ref.ref.fsdp_config.param_offload=True \
                actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
                actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
                actor_rollout_ref.rollout.name=vllm \
                actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
                actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
                actor_rollout_ref.rollout.n=${ROLLOUT_N} \
                actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE} \
                actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P} \
                actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
                actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
                actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
                actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
                actor_rollout_ref.rollout.enable_chunked_prefill=False \
                actor_rollout_ref.rollout.free_cache_engine=True \
                actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024 \
                trainer.balance_batch=True \
                trainer.logger='["console"]' \
                trainer.project_name=${PROJECT_NAME} \
                trainer.experiment_name=${EXPERIMENT_NAME} \
                trainer.n_gpus_per_node=${NTRAINER_GPUS_PER_NODE} \
                trainer.nnodes=${NNODES} \
                trainer.device=npu \
                trainer.default_local_dir=${DEFAULT_LOCAL_DIR} \
                trainer.val_before_train=True \
                trainer.save_freq=${SAVE_FREQ} \
                trainer.test_freq=${VAL_FREQ} \
                trainer.total_epochs=${TOTAL_EPOCHS} >> ${LOG_PATH}/${SAVELOG_FILE_NAME} 2>&1
            break
        fi

        echo "Waiting for Ray nodes. Current=${device_count}, expected=${NNODES}"
        sleep 10
    done
else
    # ---- 工作节点：注册到 Ray Head ----
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
