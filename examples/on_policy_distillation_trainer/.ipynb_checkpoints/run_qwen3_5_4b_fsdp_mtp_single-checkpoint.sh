#!/bin/bash

# On-policy distillation | vision (Qwen3.5) | vLLM rollout | FSDP training | Ascend NPU
# 单节点 16 卡版本：1 机 16 NPU，Student 和 Teacher 各占 8 卡

# dependency: NPU vllm==0.18.0, vllm-ascend@<54879467>, transformers@<cc7ab9be>

echo "***************************************** copy file *****************************************"
WORKDIR=`pwd`
echo "inital dictionary: $WORKDIR"
PROJECT_ROOT=/cache/algorithm

ROOT_PATH="/opt/huawei"
cp -r ${ROOT_PATH}/schedule-train/algorithm ${PROJECT_ROOT}
ls ${PROJECT_ROOT}

# /home/ma-user/work/dataset/cq_qwen_datasets/tools/cann_packages/cann9.0.0
cp ${cq_qwen_datasets}/tools/cann_packages/cann9.0.0/Ascend-cann-toolkit_9.0.0_linux-aarch64.run ${PROJECT_ROOT}
cp ${cq_qwen_datasets}/tools/cann_packages/cann9.0.0/Ascend-cann-A3-ops_9.0.0_linux-aarch64.run ${PROJECT_ROOT}
cp ${cq_qwen_datasets}/tools/cann_packages/cann9.0.0/triton_ascend-3.2.1-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl ${PROJECT_ROOT}
# cp ${cq_qwen_datasets}/tools/cann_packages/cann851/torch_npu-2.7.1.post2-cp311-cp311-manylinux_2_28_aarch64.whl ${PROJECT_ROOT}
# cp ${cq_qwen_datasets}/algorithm/env_verl/tools/cannb030/torch-2.7.1-cp311-cp311-manylinux_2_28_aarch64.whl ${PROJECT_ROOT}

ls ${PROJECT_ROOT}
echo "***************************************** update cann *****************************************"
# source ${cq_qwen_datasets}cann_env/ascend-toolkit/set_env.sh
# source ${cq_qwen_datasets}cann_env/nnal/atb/set_env.sh

cd ${PROJECT_ROOT}
chmod +777 ./Ascend-cann-*
./Ascend-cann-toolkit_9.0.0_linux-aarch64.run --install --quiet
source /home/ma-user/Ascend/cann/set_env.sh
./Ascend-cann-A3-ops_9.0.0_linux-aarch64.run --install --quiet
source /home/ma-user/Ascend/cann/set_env.sh

export LD_LIBRARY_PATH=/home/ma-user/Ascend/cann-9.0.0/aarch64-linux/devlib/linux/aarch64:$LD_LIBRARY_PATH
echo "--------------------- adapt MTP end!-----------------------"
export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
# export TASK_QUEUE_ENABLE=2
# vLLM NPU graph capture does not support TASK_QUEUE_ENABLE=2
export TASK_QUEUE_ENABLE=1
# export ASCEND_LAUNCH_BLOCKING=0
# Debug only: make NPU errors synchronous so stacktrace points to the real failing op
export ASCEND_LAUNCH_BLOCKING=1
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1
# export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# vLLM memory pool is incompatible with expandable_segments
unset PYTORCH_NPU_ALLOC_CONF

export ACL_OP_COMPILER_CACHE_DIR=$ACL_OP_COMPILER_CACHE_DIR
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_EXEC_TIMEOUT=7200
export HCCL_IF_BASE_PORT=64000
export ACL_DEVICE_SYNC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export TORCH_DIST_TIMEOUT=3600
echo "***************************************** Installing Pytorch *****************************************"
python3 -m pip install --upgrade pip
pip uninstall jsonnet
pip install --upgrade pip

cd ${PROJECT_ROOT}
# TransferQueue==0.1.8
pip install TransferQueue
pip install transferqueue-0.1.9.dev0-py3-none-any.whl
pip install torch==2.9.0
pip install torch_npu==2.9.0.post2
pip uninstall torch_c_dlpack_ext
pip install triton_ascend-3.2.1-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
# pip install torch_npu-2.7.1.post2-cp311-cp311-manylinux_2_28_aarch64.whl
pip install torch==2.9.0
pip install torch_npu==2.9.0.post2
python -c "import torch;import torch_npu;print(torch_npu.npu.is_available())"

echo "***************************************** source ATB / torch libs *****************************************"
TORCH_LIB_DIR=$(python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')")
ATB_LIB_DIR=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib
export LD_LIBRARY_PATH=${TORCH_LIB_DIR}:${ATB_LIB_DIR}:$LD_LIBRARY_PATH
echo "TORCH_LIB_DIR=${TORCH_LIB_DIR}"
echo "ATB_LIB_DIR=${ATB_LIB_DIR}"

ldd /home/ma-user/anaconda3/envs/PyTorch-2.7.1/lib/python3.11/site-packages/torch_npu/lib/libop_plugin_atb.so | grep -i "not found\|atb\|torch\|c10" || true


echo "*****************************************Start install transformers/vllm/vllm-ascend*****************************************"
cd ${PROJECT_ROOT}
pip install -r requirements-npu.txt
pip install tokenizers==0.21.1 transformers==5.3.0 accelerate==1.2.0

pip install vllm==0.18.0
# pip install vllm-ascend==0.18.0
# pip uninstall -y vllm-ascend vllm_ascend
pip install /opt/huawei/dataset/wyb_qwen_small/install_sourse/wheels/vllm_ascend*.wh
# pip install /opt/huawei/dataset/wyb_qwen_small/install_sourse/wheels/vllm_ascend*.whl
pip install triton_ascend-3.2.1-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
python - <<'PY'
from vllm_ascend import _build_info
from vllm_ascend.utils import check_ascend_device_type
print("vllm_ascend device:", _build_info.__device_type__)
check_ascend_device_type()
print("vllm_ascend device check ok")
PY


pip install -U hydra-core codetiming

# ---- user-adjustable ----
STUDENT_MODEL=${STUDENT_MODEL:-/opt/huawei/dataset/wyb_qwen_small/Qwen3.5-2B}
TEACHER_MODEL=${TEACHER_MODEL:-/opt/huawei/dataset/wyb_qwen_small/Qwen3.5-4B}

# 单节点配置
NNODES=1
NGPUS_PER_NODE=${MA_NUM_GPUS:-16}

# 单节点 16 卡资源分配：Student 和 Teacher 各占一半
# Student (global_pool): 8卡 (FSDP训练 + vLLM rollout)
# Teacher (teacher_pool): 8卡 (vLLM推理)
STUDENT_GPUS_PER_NODE=${STUDENT_GPUS_PER_NODE:-8}
TEACHER_GPUS_PER_NODE=${TEACHER_GPUS_PER_NODE:-8}

distillation_loss_mode=${DISTILLATION_LOSS_MODE:-k1}
use_policy_gradient=${USE_POLICY_GRADIENT:-True}
distillation_topk=${DISTILLATION_TOPK:-64}

train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-128}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

actor_lr=${ACTOR_LR:-1e-6}

# Student rollout: TP=2, 在8卡上可运行4个rollout组
rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}
# Teacher inference: TP=2, 8卡上可运行4个teacher副本
# Qwen3.5-4B 是 Dense 模型，不需要 expert_parallel_size
teacher_tp=${TEACHER_TP:-2}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.4}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-200}
test_freq=${TEST_FREQ:-5}

project_name=${PROJECT_NAME:-verl_distill_geo3k}
experiment_name=${EXPERIMENT_NAME:-qwen3_5_4b_single_node_vllm_fsdp}

# # 数据集 (Geo3K multimodal, parquet 含 images 列)
# geo3k_train=/opt/huawei/dataset/Geometry3K/geo3k/train.parquet
# geo3k_test=/opt/huawei/dataset/Geometry3K/geo3k/test.parquet

# train_files="['$geo3k_train']"
# val_files="['$geo3k_test']"

# 数据集路径
gsm8k_train=/opt/huawei/dataset/zyr_yuyin/youshen/datasets/gsm8k_train.parquet
gsm8k_test=/opt/huawei/dataset/zyr_yuyin/youshen/datasets/gsm8k_test.parquet

train_files="['$gsm8k_train']"
val_files="['$gsm8k_test']"

default_local_dir=/opt/huawei/dataset/cq_qwen_datasets/opd_test

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))


### 启动分布式网络配置（单节点）
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1

export MASTER_PORT=6167
export DASHBOARD_PORT=8191

export TMPDIR="/cache/ray_tmp"
mkdir -p /cache/ray_tmp

export NPUS_PER_NODE=$NGPUS_PER_NODE
export WORLD_SIZE=$NPUS_PER_NODE

readonly SOCKET_IFNAME="eth0"
export CURRENT_IP=127.0.0.1

pip install -U "ray"

pip install numpy av ftfy bs4 beartype pandas==2.0.3 diffusers==0.30.3 accelerate==0.32.1 pandarallel datasets qwen_vl_utils imageio imageio-ffmpeg jsonargparse peft==0.7.1 pydantic>=2.10.4 gpytorch==1.14

pip install jsonnet==0.17.0
pip install torchdata rich
pip install tensordict

# cd ${PROJECT_ROOT}/vllm-ascend
# export COMPILE_CUSTOM_KERNELS=1
# pip install -r requirements.txt
# pip install -v -e . --no-build-isolation

# ================= Ray 资源清理与环境就绪 =================
echo "Cleaning up old Ray processes..."
ray stop --force
sleep 5
rm -rf /cache/ray_tmp/*
pkill -9 -f raylet || true
pkill -9 -f plasma_store || true
pkill -9 -f gcs_server || true
echo "Waiting 20s for NPU to release resources..."

# pip install datasets==2.21.0

pip install flash-linear-attention==0.5.0

cd ${PROJECT_ROOT}
pip install "datasets==4.3.0"
pip install triton_ascend-3.2.1-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
# pip install -U triton
# pip install triton-ascend==3.2.1
# pip install triton-ascend==3.2.0
pip install -U hydra-core

pip uninstall torch_c_dlpack_ext -y

pip install torch==2.9.0
pip install torch_npu==2.9.0.post2

pip list

export RAY_DEDUP_LOGS=0
export HCCL_SOCKET_IFNAME=$SOCKET_IFNAME
export GLOO_SOCKET_IFNAME=$SOCKET_IFNAME
export TP_SOCKET_IFNAME=$SOCKET_IFNAME
export ASCEND_GLOBAL_LOG_LEVEL=3
# export ASCEND_PROCESS_LOG_PATH=/opt/huawei/dataset/zyr_yuyin/r
export HYDRA_FULL_ERROR=1
export RAY_gcs_server_rpc_server_thread_num=32
export RAY_gcs_server_request_timeout_seconds=600
export RAY_timeout_ms=600000
export RAY_worker_register_timeout_seconds=600
export RAY_USAGE_STATS_ENABLED=0

# transformers==5.3.0
pip install transformers==5.3.0 accelerate==1.2.0
# python - <<'PY'
# import pathlib, transformers

# p = pathlib.Path(transformers.__file__).parent / "modeling_rope_utils.py"
# text = p.read_text()
# old = 'ignore_keys_at_rope_validation = ignore_keys_at_rope_validation | {"partial_rotary_factor"}'
# new = 'ignore_keys_at_rope_validation = set(ignore_keys_at_rope_validation) | {"partial_rotary_factor"}'

# if old not in text and new not in text:
#     raise RuntimeError(f"target line not found in {p}")

# if new not in text:
#     p.write_text(text.replace(old, new))

# print("patched rope utils:", p)
# PY
# python - <<'PY'
# from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config
# print("vllm qwen3_5 config ok")
# PY
pip list

# ================= 单节点：启动 Ray Head 并直接运行训练 =================
echo "Starting Ray HEAD node (single-node mode) on 127.0.0.1..."
ray start --head \
    --node-ip-address=127.0.0.1 \
    --port=$MASTER_PORT \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=$DASHBOARD_PORT \
    --resources='{"NPU": '$NPUS_PER_NODE'}' \
    --block &

echo "Waiting 30s for Ray GCS to initialize..."
sleep 30

# 设置 RAY_ADDRESS 环境变量，使 ray CLI 工具无需 Dashboard 即可工作
export RAY_ADDRESS="127.0.0.1:$MASTER_PORT"

# ---------- 确认 Ray 集群资源 ----------
echo "Checking Ray cluster resources..."
ray_status_output=$(ray status 2>&1)
current_total_npus=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | awk '{s+=$1} END {print int(s)}')
if [ -z "$current_total_npus" ]; then current_total_npus=0; fi
echo "Resources: Found $current_total_npus NPUs (expected $NPUS_PER_NODE)"

if [ "$current_total_npus" -lt "$NPUS_PER_NODE" ]; then
    echo "WARNING: Not all NPUs detected. Waiting up to 60s..."
    for i in $(seq 1 12); do
        sleep 5
        ray_status_output=$(ray status 2>&1)
        current_total_npus=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | awk '{s+=$1} END {print int(s)}')
        if [ -n "$current_total_npus" ] && [ "$current_total_npus" -ge "$NPUS_PER_NODE" ]; then
            echo "All $current_total_npus NPUs detected!"
            break
        fi
        echo "  retry $i/12: found $current_total_npus NPUs..."
    done
fi

# ---------- 直接运行 OPD 蒸馏任务（绕过 ray job submit，不依赖 Dashboard） ----------
# 先通过 Python 调用 ray.init(address="auto") 连接到已启动的 Ray 集群，
# 这样 main_ppo.py 检测到 ray.is_initialized() 为 True 会跳过重复初始化。
# runtime_env 的环境变量已在 shell 中 export，无需通过 ray job submit 传递。
echo "Connecting to Ray cluster via ray.init(address='auto')..."
python3 -c "import ray; ray.init(address='auto'); print('Ray connected! Nodes:', len(ray.nodes()))"

echo "Launching OPD training directly (single-node, 16 NPU)..."
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path="$STUDENT_MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    trainer.balance_batch=True \
    trainer.logger='["console"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${STUDENT_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.device=npu \
    trainer.default_local_dir=${default_local_dir} \
    trainer.val_before_train=False \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    distillation.enabled=True \
    distillation.n_gpus_per_node=${TEACHER_GPUS_PER_NODE} \
    distillation.nnodes=${NNODES} \
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL" \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp} \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util} \
    distillation.teacher_models.teacher_model.inference.max_model_len=${max_num_tokens} \
    distillation.distillation_loss.loss_mode=${distillation_loss_mode} \
    distillation.distillation_loss.topk=${distillation_topk} \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient} \
    distillation.distillation_loss.loss_max_clamp=10.0 \
    distillation.distillation_loss.log_prob_min_clamp=-10.0

echo "Training completed."