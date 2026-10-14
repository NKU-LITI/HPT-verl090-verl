#!/bin/bash

# On-policy distillation | vision (Qwen3.5) | vLLM rollout | FSDP training | Ascend NPU/GPU

# dependency: NPU vllm==0.18.0, vllm-ascend@<54879467>, transformers@<cc7ab9be>

echo "***************************************** copy file *****************************************"
WORKDIR=`pwd`
echo "inital dictionary: $WORKDIR"
PROJECT_ROOT=/cache/algorithm

ROOT_PATH="/opt/huawei"
cp -r ${ROOT_PATH}/schedule-train/algorithm ${PROJECT_ROOT}
ls ${PROJECT_ROOT}

cp ${cq_qwen_datasets}/tools/cann_packages/cann852/Ascend-cann-toolkit_8.5.2_linux-aarch64.run ${PROJECT_ROOT}
cp ${cq_qwen_datasets}/tools/cann_packages/cann852/Ascend-cann-A3-ops_8.5.2_linux-aarch64.run ${PROJECT_ROOT}
cp ${cq_qwen_datasets}/tools/cann_packages/cann851/torch_npu-2.7.1.post2-cp311-cp311-manylinux_2_28_aarch64.whl ${PROJECT_ROOT}
cp ${cq_qwen_datasets}/algorithm/env_verl/tools/cannb030/torch-2.7.1-cp311-cp311-manylinux_2_28_aarch64.whl ${PROJECT_ROOT}

ls ${PROJECT_ROOT}
echo "***************************************** update cann *****************************************"
# source ${cq_qwen_datasets}cann_env/ascend-toolkit/set_env.sh
# source ${cq_qwen_datasets}cann_env/nnal/atb/set_env.sh

cd ${PROJECT_ROOT}
chmod +777 ./Ascend-cann-*
./Ascend-cann-toolkit_8.5.2_linux-aarch64.run --install --quiet
source /home/ma-user/Ascend/cann-8.5.2/set_env.sh
./Ascend-cann-A3-ops_8.5.2_linux-aarch64.run --install --quiet
source /home/ma-user/Ascend/cann-8.5.2/set_env.sh

echo "--------------------- adapt MTP end!-----------------------"
export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

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
pip install torch-2.7.1-cp311-cp311-manylinux_2_28_aarch64.whl
pip install torch_npu-2.7.1.post2-cp311-cp311-manylinux_2_28_aarch64.whl

python -c "import torch;import torch_npu;print(torch_npu.npu.is_available())"

echo "*****************************************Start install transformers\vllm\vllm-ascend*****************************************"
cd ${PROJECT_ROOT}
pip install -r requirements-npu.txt
pip3 install -r requirements-npu.txt

pip install tokenizers==0.21.1
pip install transformers==5.2.0 accelerate==1.2.0

pip install vllm==0.18.0
pip install vllm-ascend==0.18.0


pip install -U hydra-core codetiming

# ---- user-adjustable ----
STUDENT_MODEL=${STUDENT_MODEL:-/opt/huawei/dataset/wyb_qwen_small/Qwen3.5-2B}
TEACHER_MODEL=${TEACHER_MODEL:-/opt/huawei/dataset/wyb_qwen_small/Qwen3.5-4B}

NNODES=${MA_NUM_HOSTS:-2}
NGPUS_PER_NODE=${MA_NUM_GPUS:-16}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-16}

distillation_loss_mode=${DISTILLATION_LOSS_MODE:-k1}
use_policy_gradient=${USE_POLICY_GRADIENT:-True}
distillation_topk=${DISTILLATION_TOPK:-64}

train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-128}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

actor_lr=${ACTOR_LR:-1e-6}

# TODO确认下
rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}
teacher_tp=${TEACHER_TP:-2}
teacher_ep=${TEACHER_EP:-8}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.4}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-200}
test_freq=${TEST_FREQ:-5}

project_name=${PROJECT_NAME:-verl_distill_geo3k}
experiment_name=${EXPERIMENT_NAME:-qwen3_5_4b_from_qwen3_5_35b_vllm_fsdp}

# # 数据集路径
# gsm8k_train=/opt/huawei/dataset/zyr_yuyin/youshen/datasets/gsm8k_train.parquet
# gsm8k_test=/opt/huawei/dataset/zyr_yuyin/youshen/datasets/gsm8k_test.parquet

# train_files="['$gsm8k_train']"
# val_files="['$gsm8k_test']"

# 数据集 (Geo3K multimodal, parquet 含 images 列)
geo3k_train=/opt/huawei/dataset/Geometry3K/geo3k/train.parquet
geo3k_test=/opt/huawei/dataset/Geometry3K/geo3k/test.parquet

train_files="['$geo3k_train']"
val_files="['$geo3k_test']"

default_local_dir=/opt/huawei/dataset/cq_qwen_datasets/opd_test

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))


### 启动分布式网络配置
export NODE_RANK=${VC_TASK_INDEX}
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
echo $MASTER_ADDR


export MASTER_PORT=6167
export DASHBOARD_PORT=8191

export TMPDIR="/cache/ray_tmp"
mkdir -p /cache/ray_tmp


export NPUS_PER_NODE=$MA_NUM_GPUS
export WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

readonly SOCKET_IFNAME="eth0"
export CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')

pip install -U "ray"

pip install numpy av ftfy bs4 beartype pandas==2.0.3 diffusers==0.30.3 accelerate==0.32.1 pandarallel datasets qwen_vl_utils imageio imageio-ffmpeg jsonargparse peft==0.7.1 pydantic>=2.10.4 gpytorch==1.14

pip install jsonnet==0.17.0
pip install torchdata rich
pip install tensordict

# ================= Ray 资源清理与环境就绪 =================
echo "Cleaning up old Ray processes..."
ray stop --force
sleep 5
rm -rf /cache/ray_tmp/*
pkill -9 -f raylet || true
pkill -9 -f plasma_store || true
pkill -9 -f gcs_server || true
echo "Waiting 20s for NPU to release resources..."

pip list
sleep 20
pip install -U hydra-core

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

pip list
# ================= Ray 节点启动及任务下发 =================
if [ "$NODE_RANK" -eq 0 ]; then
    
    # ---------- 主节点 (Head Node) ----------
    echo "【防守型策略】主节点休眠 300 秒，等待所有节点物理环境就绪..."
    sleep 300
    
    echo "Starting Ray HEAD node on $MA_CURRENT_IP..."
    # 显式绑定 node-ip-address 为 MA_CURRENT_IP，确保多机通信正常
    ray start --head \
        --node-ip-address=$MA_CURRENT_IP \
        --port=$MASTER_PORT \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=$DASHBOARD_PORT \
        --resources='{"NPU": '$NPUS_PER_NODE'}' \
        --block &

    echo "Waiting 30s for Ray GCS and Dashboard to initialize..."
    sleep 30

    # ---------- 等待 Dashboard 就绪 ----------
    echo "Checking Ray Dashboard readiness on port $DASHBOARD_PORT..."
    dashboard_ready=false
    for i in $(seq 1 30); do
        if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$DASHBOARD_PORT/api/version" 2>/dev/null | grep -q "200"; then
            echo "Ray Dashboard is READY!"
            dashboard_ready=true
            break
        else
            echo "Dashboard not ready yet... retry $i/30 (waiting 5s)"
            sleep 5
        fi
    done
    if [ "$dashboard_ready" = false ]; then
        echo "WARNING: Dashboard not ready after 150s, but continuing anyway..."
    fi
    
    echo "Waiting for worker nodes to join..."
    target_npus=$((NNODES * NPUS_PER_NODE))
    
    while true; do
        ray_status_output=$(ray status 2>&1)
        current_total_npus=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | awk '{s+=$1} END {print int(s)}')
        if [ -z "$current_total_npus" ]; then current_total_npus=0; fi
        
        echo "Check Resources: Found $current_total_npus / $target_npus NPUs"

        if [ "$current_total_npus" -ge "$target_npus" ]; then
            echo "Ray cluster is READY! (Found $current_total_npus NPUs)."
            break
        else
            echo "Waiting for workers... Retrying in 5s..."
            sleep 5
        fi
    done
    
    # ---------- 提交 OPD 蒸馏任务 ----------
    ray job submit \
        --address="http://127.0.0.1:$DASHBOARD_PORT" \
        --runtime-env=verl/trainer/runtime_env.yaml \
        -- \
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
        trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
        trainer.nnodes=${NNODES} \
        trainer.device=npu \
        trainer.default_local_dir=${default_local_dir} \
        trainer.val_before_train=False \
        trainer.save_freq=${save_freq} \
        trainer.test_freq=${test_freq} \
        trainer.total_epochs=${total_epochs} \
        distillation.enabled=True \
        distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE} \
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

    echo "Job submitted. Check Dashboard at http://$CURRENT_IP:$DASHBOARD_PORT"
    exit 0
else
    # ---------- 从节点 (Worker Node) ----------
    # 先等待 Head 节点的 GCS 端口可达，避免 ray start 内部长时间超时
    echo "Waiting for Ray Head GCS port at $MASTER_ADDR:6167 to become reachable..."
    while true; do
        # 优先使用 /dev/tcp，不可用时 fallback 到 python
        if timeout 3 bash -c "echo > /dev/tcp/$MASTER_ADDR/6167" 2>/dev/null; then
            echo "GCS port $MASTER_ADDR:6167 is reachable!"
            break
        elif python3 -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(3)
try:
    s.connect(('$MASTER_ADDR', 6167))
    s.close()
    sys.exit(0)
except:
    sys.exit(1)
" 2>/dev/null; then
            echo "GCS port $MASTER_ADDR:6167 is reachable (via python)!"
            break
        else
            echo "GCS port not reachable yet, retrying in 10s..."
            sleep 10
        fi
    done

    # 端口可达后再额外等待，确保 GCS 完全初始化
    sleep 15

    # 连接 Ray 集群，带重试机制
    max_retries=5
    for attempt in $(seq 1 $max_retries); do
        echo "Connecting to Ray cluster at $MASTER_ADDR:6167 from $MA_CURRENT_IP (attempt $attempt/$max_retries)..."
        ray start --address="$MASTER_ADDR:6167" \
            --node-ip-address=$MA_CURRENT_IP \
            --resources='{"NPU": '$NPUS_PER_NODE'}' \
            --block &
        ray_pid=$!

        # 等待几秒检查进程是否还在运行
        sleep 10
        if kill -0 $ray_pid 2>/dev/null; then
            echo "Ray worker is running (PID=$ray_pid). Keeping process alive..."
            wait $ray_pid
            exit 0
        else
            echo "Ray worker process exited unexpectedly. Cleaning up and retrying..."
            ray stop --force 2>/dev/null || true
            sleep 5
        fi
    done
    echo "Failed to join Ray cluster after $max_retries attempts."
    exit 1
fi
