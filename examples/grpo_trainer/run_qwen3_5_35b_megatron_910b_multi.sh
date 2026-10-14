#!/bin/bash

# GRPO | Qwen3.5-35B-A3B MoE | Geo3K | vLLM rollout | Megatron | Ascend 910B multi-node
# 多节点版本：Megatron 引擎，纯 GRPO，4 节点 × 8 NPU
#
# Requirements on Ascend:
#   - 32 NPUs (4 nodes × 8 NPUs each, e.g. 4×8 Atlas A3)
#   - Megatron-LM==0.16.1
#   - MindSpeed==0.16.0
#   - Megatron-Bridge==de93536e
#
# Qwen3.5 architecture notes:
#   Qwen3.5 uses Gated Delta Net (GDN) linear attention which currently does
#   NOT support packed sequences (THD format) in Megatron-LM. Therefore:
#     - model.use_remove_padding=False
#     - actor.megatron.use_remove_padding=False  (forces bshd compute format)
#     - actor.use_dynamic_bsz=False              (required for bshd mode)
#
#   Once Megatron-LM adds THD support for Qwen3.5 GDN, use_remove_padding
#   can be set to True for better performance.
#
# Parallelism config (32 GPUs / 4 nodes):
#   TP=2 PP=4 CP=1 EP=4 ETP=1 GEN_TP=8

echo "***************************************** copy file *****************************************"
WORKDIR=`pwd`
echo "inital dictionary: $WORKDIR"
export PROJECT_ROOT=/cache/algorithm

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
# NOTE: NO NON_MEGATRON=true — this is a Megatron script, not FSDP
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=0
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

# vLLM / Megatron env
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

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

########################### User-adjustable Config ###########################

# ---- Megatron parallelism for 4 nodes × 8 NPUs = 32 devices ----
TP=${TP:-2}
PP=${PP:-4}
CP=${CP:-1}
EP=${EP:-4}
ETP=${ETP:-1}
GEN_TP=${GEN_TP:-8}

ALL_OFFLOAD=${ALL_OFFLOAD:-True}

# ---- Model & Data paths ----
MODEL_PATH=${MODEL_PATH:-${cq_qwen_datasets}/opensource_multimodal_models_guiyang/Qwen/Qwen3.5-35B-A3B/Qwen/Qwen3___5-35B-A3B}

TRAIN_FILE=${TRAIN_FILE:-/opt/huawei/dataset/Geometry3K/geo3k/train.parquet}
TEST_FILE=${TEST_FILE:-/opt/huawei/dataset/Geometry3K/geo3k/test.parquet}

# ---- Multi-node config ----
NNODES=${NNODES:-4}
NGPUS_PER_NODE=${MA_NUM_GPUS:-8}
NTRAINER_GPUS_PER_NODE=${NTRAINER_GPUS_PER_NODE:-8}

# ---- Training hyperparameters (aligned with working FSDP script) ----
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}

ACTOR_LR=${ACTOR_LR:-1e-6}
PPO_EPOCHS=${PPO_EPOCHS:-1}
CLIP_RATIO=${CLIP_RATIO:-0.2}
GRAD_CLIP=${GRAD_CLIP:-1.0}

ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.4}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.99}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
SAVE_FREQ=${SAVE_FREQ:-20}
VAL_FREQ=${VAL_FREQ:-10}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_geo3k}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_5_35b_megatron_geo3k_grpo_910b_multi}

RUN_TAG=${RUN_TAG:-qwen3_5_35b-megatron-geo3k-910b-multi-n${ROLLOUT_N}-bs${TRAIN_BATCH_SIZE}-mbs${PPO_MINI_BATCH_SIZE}-lr${ACTOR_LR}-$(date +%Y%m%d_%H%M%S)}

DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-/opt/huawei/dataset/wyb_qwen_small/checkpoints_35b_megatron/${RUN_TAG}}
LOG_PATH=${LOG_PATH:-/opt/huawei/dataset/wyb_qwen_small/logs/GRPO_35B_MEGATRON/${RUN_TAG}}
SAVELOG_FILE_NAME=${SAVELOG_FILE_NAME:-train.log}

# ---- end user-adjustable ----

########################### Parameter Arrays ###########################

DATA=(
    data.train_files="['${TRAIN_FILE}']"
    data.val_files="['${TEST_FILE}']"
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
)

ACTOR=(
    actor_rollout_ref.actor.strategy=megatron
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO}
    # actor_rollout_ref.actor.grad_clip=${GRAD_CLIP}
    actor_rollout_ref.actor.optim.clip_grad=${GRAD_CLIP}
    actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.calculate_entropy=False
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True

    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.use_remove_padding=False
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.dtype=bfloat16

    actor_rollout_ref.actor.checkpoint.strict=False
    actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}
    actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P}
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
)

REF=(
    actor_rollout_ref.ref.strategy=megatron
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}

    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.ref.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.ref.megatron.use_mbridge=True
    actor_rollout_ref.ref.megatron.vanilla_mbridge=False
    actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
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
    trainer.test_freq=${VAL_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
)

EXTRA=(
    model_engine=megatron
)

########################### Multi-node Network Config ###########################

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

pip install -U "ray"

pip install numpy av ftfy bs4 beartype pandas==2.0.3 diffusers==0.30.3 accelerate==1.13.0 pandarallel datasets qwen_vl_utils imageio imageio-ffmpeg jsonargparse peft==0.7.1 "pydantic>=2.10.4" gpytorch==1.14

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

pip install flash-linear-attention==0.5.0

cd ${PROJECT_ROOT}
pip install "datasets==4.3.0"
pip install numpy==1.26.4
pip install triton-ascend==3.2.0
pip install -U hydra-core

pip uninstall torch_c_dlpack_ext -y || true
pip install torch==2.9.0
pip install torch_npu==2.9.0

echo "***************************************** install Megatron-LM / MindSpeed / mbridge *****************************************"
cd ${PROJECT_ROOT}

if [ -d "${PROJECT_ROOT}/Megatron-LM" ]; then
    echo "Installing Megatron-LM from ${PROJECT_ROOT}/Megatron-LM..."
    pip install -e ${PROJECT_ROOT}/Megatron-LM
    export PYTHONPATH=${PROJECT_ROOT}/Megatron-LM:${PYTHONPATH}
elif [ -d "${PROJECT_ROOT}/megatron" ]; then
    echo "Megatron-LM source directory not found, but ${PROJECT_ROOT}/megatron exists."
    echo "Using existing megatron package via PYTHONPATH."
    export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH}
elif python -c "from megatron.core import parallel_state" 2>/dev/null; then
    echo "Megatron-LM already importable"
else
    echo "FATAL: Megatron-LM not found. Expected ${PROJECT_ROOT}/Megatron-LM or an importable megatron package."
    exit 1
fi

if [ -d "${PROJECT_ROOT}/MindSpeed" ]; then
    echo "Installing MindSpeed from ${PROJECT_ROOT}/MindSpeed..."
    rm -rf /cache/MindSpeed
    cp -r ${PROJECT_ROOT}/MindSpeed /cache/MindSpeed
    cd /cache/MindSpeed
    pip install -e .
    cd ${PROJECT_ROOT}
elif python -c "import mindspeed" 2>/dev/null; then
    echo "MindSpeed already importable"
else
    echo "FATAL: MindSpeed not found. Expected ${PROJECT_ROOT}/MindSpeed or an importable mindspeed package."
    exit 1
fi

python - <<'PY'
import os
import importlib.util
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"])
target_paths = []

source_path = project_root / "Megatron-LM" / "megatron" / "core" / "ssm" / "mamba_hybrid_layer_allocation.py"
if source_path.exists():
    target_paths.append(source_path)

spec = importlib.util.find_spec("megatron.core.ssm.mamba_hybrid_layer_allocation")
if spec and spec.origin:
    import_path = Path(spec.origin)
    if import_path.exists():
        target_paths.append(import_path)

target_paths = list(dict.fromkeys(target_paths))
if not target_paths:
    raise RuntimeError("Cannot locate Megatron-LM mamba_hybrid_layer_allocation.py in source tree or import path")

patch_text = r'''

# VERL_PARSE_HYBRID_PATTERN_COMPAT
# Compatibility backport for Megatron-Bridge flop_utils on older Megatron-LM.
from dataclasses import dataclass as _verl_dataclass


@_verl_dataclass(frozen=True)
class ParsedHybridPattern:
    main_pattern: str
    mtp_pattern: str | None = None
    mtp_num_depths: int = 0


def _verl_validate_hybrid_pattern(pattern: str, allow_pipe: bool = False) -> None:
    allowed = set(Symbols.VALID)
    if allow_pipe:
        allowed.add("|")
    invalid = sorted(set(pattern) - allowed)
    if invalid:
        raise ValueError(f"Hybrid layer pattern contains invalid symbols: {invalid}")


def parse_hybrid_pattern(pattern: str) -> ParsedHybridPattern:
    """Parse unified hybrid layer pattern with optional MTP suffix patterns.

    Examples:
        MM -> ParsedHybridPattern(main_pattern="MM")
        MM/MM/MM -> ParsedHybridPattern(main_pattern="MM", mtp_pattern="MM", mtp_num_depths=2)
        M-M-|M-M*-/MM/MM -> ParsedHybridPattern(main_pattern="M-M-|M-M*-", mtp_pattern="MM", mtp_num_depths=2)
    """
    if not pattern:
        return ParsedHybridPattern(main_pattern="")

    parts = pattern.split("/")
    main_pattern = parts[0]
    mtp_patterns = parts[1:]

    _verl_validate_hybrid_pattern(main_pattern, allow_pipe=True)
    if not mtp_patterns:
        return ParsedHybridPattern(main_pattern=main_pattern)

    mtp_pattern = mtp_patterns[0]
    _verl_validate_hybrid_pattern(mtp_pattern, allow_pipe=True)
    for item in mtp_patterns[1:]:
        _verl_validate_hybrid_pattern(item, allow_pipe=True)
        if item != mtp_pattern:
            raise ValueError("MTP hybrid layer patterns must be identical")

    return ParsedHybridPattern(
        main_pattern=main_pattern,
        mtp_pattern=mtp_pattern,
        mtp_num_depths=len(mtp_patterns),
    )
'''

print("VERL_PATCH_MEGATRON_LM_SSM_API: targets:")
for target_path in target_paths:
    print(f"  - {target_path}")
    text = target_path.read_text()
    if "VERL_PARSE_HYBRID_PATTERN_COMPAT" not in text:
        print(f"VERL_PATCH_MEGATRON_LM_SSM_API: patching {target_path}")
        target_path.write_text(text + patch_text)
    else:
        print(f"VERL_PATCH_MEGATRON_LM_SSM_API: already available in {target_path}")

    patched_text = target_path.read_text()
    if "def parse_hybrid_pattern" not in patched_text:
        raise RuntimeError(f"Megatron-LM SSM API patch failed for {target_path}")
    print(f"VERL_PATCH_MEGATRON_LM_SSM_API: verified {target_path}")
PY

python -m pip uninstall -y transformer-engine transformer-engine-torch transformer-engine-cu12 transformer-engine-cu13 || true
python -m pip install nvidia-modelopt nvidia-ml-py nvidia-resiliency-ext megatron-energon

if [ -d "${PROJECT_ROOT}/Megatron-Bridge" ]; then
    echo "Installing Megatron-Bridge from ${PROJECT_ROOT}/Megatron-Bridge..."
    export PYTHONPATH=${PROJECT_ROOT}/Megatron-Bridge/src:${PYTHONPATH}
    python -m pip install --no-deps -e ${PROJECT_ROOT}/Megatron-Bridge
elif python -c "from megatron.bridge import AutoBridge" 2>/dev/null; then
    echo "Megatron-Bridge already importable"
else
    echo "FATAL: Megatron-Bridge not found. Expected ${PROJECT_ROOT}/Megatron-Bridge or an importable megatron.bridge package."
    exit 1
fi

python - <<'PY'
import os
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"])
train_utils_path = project_root / "Megatron-Bridge" / "src" / "megatron" / "bridge" / "training" / "utils" / "train_utils.py"
if not train_utils_path.exists():
    raise RuntimeError(f"Cannot locate Megatron-Bridge train_utils.py at {train_utils_path}")

text = train_utils_path.read_text()
if "class LinearForLastLayer" not in text:
    print(f"VERL_PATCH_MEGATRON_BRIDGE_HELPERS: patching {train_utils_path}")
    train_utils_path.write_text(
        text
        + r'''

# VERL compatibility backport from Megatron-Bridge PR #3813.
class LinearForLastLayer(nn.Linear):
    """Final replicated projection head compatible with Megatron output-layer calls."""

    def __init__(self, input_size: int, output_size: int, sequence_parallel: bool) -> None:
        super().__init__(in_features=input_size, out_features=output_size, bias=False)
        self.sequence_parallel = sequence_parallel
        if sequence_parallel:
            setattr(self.weight, "sequence_parallel", True)

    def forward(self, input_, weight=None, runtime_gather_output=None):
        del weight, runtime_gather_output
        logits = super().forward(input_).float()
        if self.sequence_parallel:
            from megatron.core import tensor_parallel as _verl_tensor_parallel

            logits = _verl_tensor_parallel.gather_from_sequence_parallel_region(
                logits,
                tensor_parallel_output_grad=False,
            )
        return logits, None


def _verl_ensure_model_list(model):
    return model if isinstance(model, list) else [model]


def _verl_freeze_parameter_if_present(module, name):
    parameter = getattr(module, name, None)
    if parameter is not None:
        parameter.requires_grad = False


def _verl_register_linear_for_last_layer_mapping():
    from megatron.bridge.models.conversion.param_mapping import AutoMapping

    AutoMapping.register_module_type("LinearForLastLayer", "replicated")


def create_value_head_hook(hidden_size: int, sequence_parallel: bool, output_size: int = 1):
    from megatron.core import parallel_state

    _verl_register_linear_for_last_layer_mapping()

    def hook(model):
        model_chunks = _verl_ensure_model_list(model)
        model_post_process = []
        if (
            parallel_state.get_pipeline_model_parallel_world_size() > 1
            and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
        ):
            for vp_stage in range(parallel_state.get_virtual_pipeline_model_parallel_world_size()):
                model_post_process.append(
                    parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
                )
        else:
            model_post_process.append(parallel_state.is_pipeline_last_stage())

        if len(model_post_process) != len(model_chunks):
            raise ValueError(
                "Model list length and pipeline post-process list length must match. "
                f"Got {len(model_chunks)} model chunks and {len(model_post_process)} post-process flags."
            )

        for index, model_chunk in enumerate(model_chunks):
            if model_post_process[index]:
                model_chunk.output_layer = LinearForLastLayer(
                    input_size=hidden_size,
                    output_size=output_size,
                    sequence_parallel=sequence_parallel,
                )

        return model_chunks

    return hook


def make_value_model(hidden_size: int, sequence_parallel: bool):
    return create_value_head_hook(hidden_size=hidden_size, sequence_parallel=sequence_parallel)


def freeze_moe_router(model):
    model_chunks = _verl_ensure_model_list(model)
    for model_chunk in model_chunks:
        decoder = getattr(model_chunk, "decoder", None)
        layers = getattr(decoder, "layers", None)
        if layers is None:
            continue
        for layer in layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            router = getattr(mlp, "router", None)
            if router is not None:
                _verl_freeze_parameter_if_present(router, "weight")
                _verl_freeze_parameter_if_present(router, "bias")

            shared_experts = getattr(mlp, "shared_experts", None)
            if shared_experts is not None:
                _verl_freeze_parameter_if_present(shared_experts, "gate_weight")
                _verl_freeze_parameter_if_present(shared_experts, "gate_bias")

    return model_chunks
'''
    )
else:
    print(f"VERL_PATCH_MEGATRON_BRIDGE_HELPERS: already available in {train_utils_path}")

patched_text = train_utils_path.read_text()
missing = [
    name
    for name in ("class LinearForLastLayer", "def make_value_model", "def freeze_moe_router")
    if name not in patched_text
]
if missing:
    raise RuntimeError(f"Megatron-Bridge helper patch failed, missing: {missing}")
print("VERL_PATCH_MEGATRON_BRIDGE_HELPERS: verified")
PY

python - <<'PY'
import os
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"])
model_provider_path = project_root / "Megatron-Bridge" / "src" / "megatron" / "bridge" / "models" / "model_provider.py"
if not model_provider_path.exists():
    raise RuntimeError(f"Cannot locate Megatron-Bridge model_provider.py at {model_provider_path}")

text = model_provider_path.read_text()
if "VERL_DYNAMIC_APPLY_OVERRIDES_AND_FINALIZE" not in text:
    print(f"VERL_PATCH_MEGATRON_BRIDGE_PROVIDER_API: patching dynamic provider API into {model_provider_path}")
    model_provider_path.write_text(
        text
        + r'''

# VERL_DYNAMIC_APPLY_OVERRIDES_AND_FINALIZE
# VERL compatibility backport from Megatron-Bridge PR #3813, adapted for older provider classes.
def _verl_apply_overrides_and_finalize(self, dtype=None, overrides=None):
    if dtype is not None:
        self.params_dtype = dtype
        self.fp16 = dtype == torch.float16
        self.bf16 = dtype == torch.bfloat16

    for name, value in (overrides or {}).items():
        if not hasattr(self, name):
            print(
                f"VERL_PATCH_MEGATRON_BRIDGE_PROVIDER_API: "
                f"set dynamic override {type(self).__name__}.{name}={value!r}"
            )
        setattr(self, name, value)

    self.finalize()
    return self


ModelProviderMixin.apply_overrides_and_finalize = _verl_apply_overrides_and_finalize
'''
    )
else:
    print(f"VERL_PATCH_MEGATRON_BRIDGE_PROVIDER_API: dynamic provider API already available in {model_provider_path}")

patched_text = model_provider_path.read_text()
if "VERL_DYNAMIC_APPLY_OVERRIDES_AND_FINALIZE" not in patched_text:
    raise RuntimeError("Megatron-Bridge provider API patch failed")
print("VERL_PATCH_MEGATRON_BRIDGE_PROVIDER_API: verified")
PY

python - <<'PY'
import os
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"])
peft_utils_path = project_root / "Megatron-Bridge" / "src" / "megatron" / "bridge" / "peft" / "utils.py"
if not peft_utils_path.exists():
    raise RuntimeError(f"Cannot locate Megatron-Bridge peft/utils.py at {peft_utils_path}")

text = peft_utils_path.read_text()
if "VERL_PEFT_EXTERNAL_TRAINER_HELPERS" not in text:
    print(f"VERL_PATCH_MEGATRON_BRIDGE_PEFT_API: patching {peft_utils_path}")
    peft_utils_path.write_text(
        text
        + r'''

# VERL_PEFT_EXTERNAL_TRAINER_HELPERS
# VERL compatibility backport from Megatron-Bridge PR #3813.
def _verl_peft_ensure_model_list(model):
    return model if isinstance(model, list) else [model]


def _verl_apply_peft(peft, model, training=True):
    transformed_model = peft(model, training=training)
    peft.set_params_to_save(transformed_model)
    return transformed_model


def create_peft_hook(peft, training=True):
    """Create a provider pre-wrap hook that applies PEFT."""

    def hook(model):
        return _verl_apply_peft(peft, model, training=training)

    return hook


def load_peft_adapter_checkpoint(
    model,
    adapter_checkpoint_path,
    peft,
    strict=False,
    model_sd_kwargs=None,
    ckpt_format="torch_dist",
    pg_collection=None,
    fully_parallel_load=True,
    load_strategy=None,
):
    """Load a PEFT adapter checkpoint into an already transformed model."""
    from megatron.core import dist_checkpointing, parallel_state as _verl_parallel_state
    from megatron.core.dist_checkpointing.serialization import get_default_load_sharded_strategy
    from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelLoadStrategyWrapper

    from megatron.bridge.training.checkpointing import (
        _generate_model_state_dict,
        apply_peft_adapter_filter_to_state_dict,
    )

    model_chunks = _verl_peft_ensure_model_list(model)
    sharded_state_dict = _generate_model_state_dict(
        model_chunks,
        model_sd_kwargs,
        ckpt_format,
        pg_collection=pg_collection,
    )
    sharded_state_dict = apply_peft_adapter_filter_to_state_dict(sharded_state_dict, peft)

    checkpoint_path = str(adapter_checkpoint_path)
    if load_strategy is None:
        load_strategy = get_default_load_sharded_strategy(checkpoint_path)
        if fully_parallel_load and _verl_parallel_state.is_initialized():
            load_strategy = FullyParallelLoadStrategyWrapper(
                load_strategy,
                _verl_parallel_state.get_data_parallel_group(with_context_parallel=True),
            )

    loaded_state_dict = dist_checkpointing.load(sharded_state_dict, checkpoint_path, load_strategy)
    for vpp_rank, model_chunk in enumerate(model_chunks):
        model_key = "model" if len(model_chunks) == 1 else f"model{vpp_rank}"
        if model_key not in loaded_state_dict:
            if len(model_chunks) == 1:
                fallback_model_key = next((key for key in loaded_state_dict if key.startswith("model")), None)
                if fallback_model_key is not None:
                    model_key = fallback_model_key
                else:
                    raise KeyError(
                        "Expected adapter checkpoint to contain a top-level 'model' or 'model*' key, "
                        f"but found keys: {list(loaded_state_dict.keys())}"
                    )
            else:
                expected_model_keys = [f"model{rank}" for rank in range(len(model_chunks))]
                raise KeyError(
                    f"Expected adapter checkpoint to contain top-level key {model_key!r} "
                    f"for virtual pipeline model chunk {vpp_rank} "
                    f"(expected keys: {expected_model_keys}), "
                    f"but found keys: {list(loaded_state_dict.keys())}"
                )
        model_chunk.load_state_dict(loaded_state_dict[model_key], strict=strict)
'''
    )
else:
    print(f"VERL_PATCH_MEGATRON_BRIDGE_PEFT_API: already available in {peft_utils_path}")

patched_text = peft_utils_path.read_text()
missing = [
    name
    for name in ("def create_peft_hook", "def load_peft_adapter_checkpoint")
    if name not in patched_text
]
if missing:
    raise RuntimeError(f"Megatron-Bridge PEFT API patch failed, missing: {missing}")
print("VERL_PATCH_MEGATRON_BRIDGE_PEFT_API: verified")
PY

python - <<'PY'
import os
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"])
config_utils_path = project_root / "Megatron-Bridge" / "src" / "megatron" / "bridge" / "training" / "utils" / "config_utils.py"
if not config_utils_path.exists():
    raise RuntimeError(f"Cannot locate Megatron-Bridge config_utils.py at {config_utils_path}")

text = config_utils_path.read_text()
if "VERL_DDP_CONFIG_HELPER" not in text:
    print(f"VERL_PATCH_MEGATRON_BRIDGE_CONFIG_API: patching {config_utils_path}")
    config_utils_path.write_text(
        text
        + r'''

# VERL_DDP_CONFIG_HELPER
# VERL compatibility backport from Megatron-Bridge PR #3813.
def create_ddp_config(
    wrap_with_ddp=True,
    use_distributed_optimizer=True,
    use_megatron_fsdp=False,
    overrides=None,
    finalize=True,
):
    """Create a finalized Bridge DDP config for external model construction."""
    if not wrap_with_ddp:
        return None

    from megatron.bridge.training.config import DistributedDataParallelConfig

    ddp_config = {
        "use_distributed_optimizer": use_distributed_optimizer,
    }
    if use_megatron_fsdp:
        ddp_config.update(
            {
                "use_distributed_optimizer": True,
                "check_for_nan_in_grad": True,
                "use_megatron_fsdp": True,
                "data_parallel_sharding_strategy": "optim_grads_params",
                "overlap_grad_reduce": True,
            }
        )
    ddp_config.update(overrides or {})

    config = DistributedDataParallelConfig(**ddp_config)
    if finalize:
        config.finalize()
    return config
'''
    )
else:
    print(f"VERL_PATCH_MEGATRON_BRIDGE_CONFIG_API: already available in {config_utils_path}")

patched_text = config_utils_path.read_text()
if "def create_ddp_config" not in patched_text:
    raise RuntimeError("Megatron-Bridge config API patch failed")
print("VERL_PATCH_MEGATRON_BRIDGE_CONFIG_API: verified")
PY

python - <<'PY'
try:
    import transformer_engine.pytorch as te
    print("Transformer Engine import ok:", te)
except Exception as exc:
    print("WARNING: transformer_engine.pytorch is not importable in this Ascend environment.")
    print("WARNING: Do not install the PyPI transformer-engine meta package here; it is CUDA/NVIDIA-oriented and breaks Megatron fallback.")
    print("WARNING: If Megatron-Bridge fails during import, use an Ascend image/package that provides TE-compatible modules via MindSpeed, or add a local compatibility shim before importing Megatron-Bridge.")
    print("WARNING: Original transformer_engine error:", repr(exc))

from megatron.core import parallel_state
import megatron.core.ssm.mamba_hybrid_layer_allocation as mamba_alloc
from megatron.core.ssm.mamba_hybrid_layer_allocation import parse_hybrid_pattern
from mindspeed.megatron_adaptor import repatch
from megatron.bridge import AutoBridge
from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.peft.utils import create_peft_hook, load_peft_adapter_checkpoint
from megatron.bridge.training.utils.config_utils import create_ddp_config
from megatron.bridge.training.utils.train_utils import LinearForLastLayer, freeze_moe_router, make_value_model
from verl.workers.engine import EngineRegistry

registry = EngineRegistry._engines
print("Megatron-LM import ok")
print("Megatron-LM SSM parser module:", mamba_alloc.__file__)
print("Megatron-LM SSM hybrid parser ok:", parse_hybrid_pattern("MM/MM/MM"))
print("MindSpeed import ok")
print("Megatron-Bridge import ok")
print("Megatron-Bridge external trainer helpers ok")
print("Megatron-Bridge PEFT helpers ok")
print("Megatron-Bridge config helpers ok")
assert hasattr(ModelProviderMixin, "apply_overrides_and_finalize")
print("Megatron-Bridge provider API ok")
print("EngineRegistry:", registry)
assert "megatron" in registry.get("language_model", {}), registry
assert "npu" in registry["language_model"]["megatron"], registry["language_model"]["megatron"]
print("VERL megatron/npu engine registration ok")
PY

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

# transformers==5.3.0
pip install mathruler pylatexenc

# 安装指定版本的transformer
pip install tokenizers==0.21.1
cd ${PROJECT_ROOT}/transformers
pip install -e .
cd ${PROJECT_ROOT}
pip list

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

            mkdir -p ${DEFAULT_LOCAL_DIR}
            mkdir -p ${LOG_PATH}

            python3 -m verl.trainer.main_ppo \
                "${DATA[@]}" \
                "${ALGORITHM[@]}" \
                "${MODEL[@]}" \
                "${ROLLOUT[@]}" \
                "${ACTOR[@]}" \
                "${REF[@]}" \
                "${TRAINER[@]}" \
                "${EXTRA[@]}" >> ${LOG_PATH}/${SAVELOG_FILE_NAME} 2>&1
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

echo "Execution finished, waiting 1 hour before exit..."
sleep 3600
