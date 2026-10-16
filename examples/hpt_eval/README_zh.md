# MTP 模型导出与 HPT Table 2 评估使用说明

本目录用于将 verl 训练产生的 SFT、DFT、GRPO checkpoint 导出为
HuggingFace 模型，并在 MTP/Ascend 环境中执行 HPT Table 2 风格评估。

> 判题默认使用本仓库 HPT/GRPO 训练相同的
> `verl.experimental.hpt.reward.compute_score`。它保证训练与离线评估的
> reward 口径一致，但不是 HPT 论文官方 DeepMath evaluator 的逐行复刻。

## 1. 确认 checkpoint

训练脚本的默认保存位置如下：

- SFT：`${zlt_data}/runs/sft/${EXP_NAME}/checkpoint/global_step_*`
- DFT：`${zlt_data}/runs/dft/${EXP_NAME}/checkpoint/global_step_*`
- GRPO：`${zlt_data}/ckpt/${PROJECT_NAME}/global_step_*/actor`

先列出实际保存的 step：

```bash
find "${zlt_data}/runs/sft" -type d -name 'global_step_*' | sort -V
find "${zlt_data}/runs/dft" -type d -name 'global_step_*' | sort -V
find "${zlt_data}/ckpt" -type d -name 'global_step_*' | sort -V
```

通常选择最终 step 或验证集表现最佳的 step。不要把 `checkpoint` 根目录直接
传给导出脚本，必须传具体的 `global_step_N` 目录。

## 2. 导出 HuggingFace 模型

在 MTP 容器的项目目录执行。SFT/DFT 传 `global_step_N`；GRPO 可以传
`global_step_N` 或 `global_step_N/actor`，脚本会自动识别 `actor`。

```bash
cd /cache/algorithm

SFT_CKPT="${zlt_data}/runs/sft/<SFT_EXP_NAME>/checkpoint/global_step_N" \
DFT_CKPT="${zlt_data}/runs/dft/<DFT_EXP_NAME>/checkpoint/global_step_N" \
GRPO_CKPT="${zlt_data}/ckpt/<GRPO_PROJECT_NAME>/global_step_N" \
MERGED_ROOT="${zlt_data}/eval_models" \
bash examples/hpt_eval/merge_checkpoints.sh
```

默认导出到：

```text
${zlt_data}/eval_models/qwen35b_sft_hf
${zlt_data}/eval_models/qwen35b_dft_hf
${zlt_data}/eval_models/qwen35b_grpo_hf
```

导出应只启动一个任务副本，不要让多个节点同时向同一个目标目录写文件。
Qwen3.5-35B-A3B 合并时会在 CPU 内存中重组 FSDP shards，应选择 CPU 内存
充足的节点，并确保 checkpoint 的全部 rank 文件都位于共享存储。

导出完成后检查：

```bash
for model_dir in \
  "${zlt_data}/eval_models/qwen35b_sft_hf" \
  "${zlt_data}/eval_models/qwen35b_dft_hf" \
  "${zlt_data}/eval_models/qwen35b_grpo_hf"
do
  test -f "${model_dir}/config.json" || echo "缺少 config.json: ${model_dir}"
  find "${model_dir}" -maxdepth 1 -name '*.safetensors' | head
done
```

如果模型已经是完整 HuggingFace 目录（含 `config.json`、tokenizer 文件和
权重文件），不需要再次导出，直接把该目录传给评估脚本即可。

## 3. 准备 Table 2 数据

默认从 `${zlt_data}/HPT_data` 读取：

```text
AIME24/test.parquet
AIME25/test.parquet
AMC23/test.parquet
MATH-500/test.parquet
Minerva/test.parquet
Olympiad-Bench/test.parquet
ARC-c/test.parquet
GPQA/test.parquet
```

原始 `HPT-verl090/data` 不含 ARC-c。本仓库已在 `data/ARC-c/test.parquet`
补齐兼容的 ARC-Challenge test split（共 1172 题）。若需要从固定 revision
的 `allenai/ai2_arc` 重新生成：

```bash
cd /cache/algorithm
python3 examples/hpt_eval/prepare_arc_c.py \
  --output "${zlt_data}/HPT_data/ARC-c/test.parquet"
```

如果环境缺少 Hugging Face datasets，先执行：

```bash
python3 -m pip install datasets -i https://pypi.tuna.tsinghua.edu.cn/simple
```

HPT 论文中的 AMC 是 AMC12 2022 与 AMC12 2023 的合集。如果已有 AMC22，
运行评估时设置：

```bash
AMC_FILES="${zlt_data}/HPT_data/AMC22/test.parquet,${zlt_data}/HPT_data/AMC23/test.parquet"
```

缺少 AMC22 时可以只使用默认 AMC23，但该列不能视为与论文 Table 2 完全
同口径。

## 4. 启动 MTP 评估

将 `examples/hpt_eval/run_table2_mtp.sh` 设置为 MTP 作业入口，建议使用与
GRPO 训练相同的节点数和每节点 8 张 910B。以下环境变量可以在平台作业
配置中设置，也可以直接写在启动命令前：

```bash
SFT_MODEL_PATH="${zlt_data}/eval_models/qwen35b_sft_hf" \
DFT_MODEL_PATH="${zlt_data}/eval_models/qwen35b_dft_hf" \
GRPO_MODEL_PATH="${zlt_data}/eval_models/qwen35b_grpo_hf" \
OUTPUT_ROOT="${zlt_data}/runs/eval_result/qwen35b_table2" \
bash examples/hpt_eval/run_table2_mtp.sh
```

如需额外评估 HPT checkpoint，先将其导出为 HuggingFace 模型，再增加：

```bash
HPT_MODEL_PATH="${zlt_data}/eval_models/qwen35b_hpt_hf"
```

如果实际数据目录名称不同，可覆盖这些变量：

```text
AIME24_FILE  AIME25_FILE  AMC_FILES  MATH500_FILE
MINERVA_FILE OLYMPIAD_FILE ARCC_FILE GPQA_FILE
```

评估默认参数与论文 Table 2 协议对齐：

- AIME24、AIME25、AMC：temperature 0.6、top-p 0.95、avg@32；
- MATH-500、Minerva、Olympiad、ARC-c、GPQA：同采样参数、avg@1；
- 最大生成长度：8192 tokens。

DFT 是额外对照行，不是原论文 Table 2 中的模型行。

## 5. 断点续跑

默认 `RESUME_GENERATION=1`。若某阶段的生成 JSON 已存在且非空，脚本会
复用它，然后重新执行打分与汇总。作业中断后使用完全相同的
`OUTPUT_ROOT` 重跑即可：

```bash
RESUME_GENERATION=1 \
OUTPUT_ROOT="${zlt_data}/evaluation/qwen35b_table2" \
bash examples/hpt_eval/run_table2_mtp.sh
```

需要重新采样全部答案时设置 `RESUME_GENERATION=0`。这会覆盖同名生成结果，
建议改用新的 `OUTPUT_ROOT`，以保留旧实验。

## 6. 结果位置

最终报告位于 `${OUTPUT_ROOT}/report/`：

- `table2.md`：便于阅读的 Table 2 风格结果；
- `table2.csv`：可直接导入表格软件；
- `token_and_truncation.csv`：各训练阶段的 token 数和截断率汇总；
- `all_benchmark_summary.csv`：阶段 × benchmark 的完整统计。

每个阶段（`SFT`、`DFT`、`GRPO`，可选 `HPT`）目录包含：

- `generated/avg32.json`、`generated/avg1.json`：原始批量生成结果；
- `rollouts.jsonl`：每条 rollout 的回答、reward、token 数、finish reason、
  格式是否合法及是否截断；
- `benchmark_summary.csv`：逐 benchmark 的准确率、token 分布、格式率和
  截断率；
- `summary.json`：该阶段的机器可读汇总。

快速查看结果：

```bash
cat "${zlt_data}/evaluation/qwen35b_table2/report/table2.md"
column -s, -t "${zlt_data}/evaluation/qwen35b_table2/report/token_and_truncation.csv" | less -S
```

## 7. 常用资源参数

显存不足时可降低 vLLM 显存比例或调整 TP：

```bash
ROLLOUT_TP=4 \
ROLLOUT_GPU_MEM_UTIL=0.50 \
bash examples/hpt_eval/run_table2_mtp.sh
```

可覆盖的主要参数还有：

```text
NNODES NPUS_PER_NODE MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH MAX_MODEL_LEN
EVAL_TEMPERATURE EVAL_TOP_P SCORER_WORKERS PYPI_INDEX_URL
```

`ROLLOUT_TP` 必须能整除模型部署所使用的设备数。修改温度、top-p、采样数
或最大生成长度后，结果将不再与 Table 2 协议同口径。
