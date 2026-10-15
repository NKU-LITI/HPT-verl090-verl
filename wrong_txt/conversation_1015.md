# HPT Qwen3.5 MTP 排错对话记录

日期：2026-10-15  
工作目录：`SFT-RL/HPT-verl090-verl/HPT-verl090-verl`

> 本文导出用户与助手之间的对话要点，不包含系统指令、内部推理和工具原始输出。

---

## 1. 初始问题：GRPO 正常，HPT 报错

### 用户

我在MTP平台执行SFT-RL/HPT-verl090-verl/HPT-verl090-verl/examples/grpo_trainer/zlt_run_grpo_qwen35b_v3.sh正常，但是SFT-RL/HPT-verl090-verl/HPT-verl090-verl/examples/grpo_trainer/zlt_run_hpt_qwen35b_v3.sh报错，SFT-RL/HPT-verl090-verl/HPT-verl090-verl/wrong_txt/1014.txt

### 助手

已定位并修复。

根因是 Qwen3.5 即使处理纯文本，也会生成 `[batch, rope_dim, seq_len]` 的三维 `position_ids`。HPT 原先简单地将所有三维位置编码判断为多模态输入，因此抛出：

```text
HPT currently supports text-only rollouts; multimodal position_ids were found
```

修改内容：

- `verl/experimental/hpt/algorithm.py`
  - 支持 Qwen3.5 纯文本多通道 RoPE。
  - 重建 HPT target 的位置编码时保留 RoPE 通道数。
  - 如果各 RoPE 通道确实不同，仍按真实多模态输入拒绝处理。
- `verl/experimental/hpt/tests/test_algorithm.py`
  - 添加 Qwen3.5 四通道纯文本位置编码回归测试。

已通过 Python 语法检查、Shell 脚本语法检查和 `git diff --check`。当前本地环境没有安装 `torch/pytest`，所以无法实际运行单测。将修改后的代码同步到 MTP 上 traceback 中的 `/opt/huawei/schedule-train/algorithm` 后，重新运行 `zlt_run_hpt_qwen35b_v3.sh` 即可验证，无需修改 HPT 启动参数。

---

## 2. MTP 强制使用远程仓库代码

### 用户

mtp平台我希望直接用远程的实验进行覆盖，就是如果发生冲突就直接用仓库里的代码，那么mtp该运行什么命令呢

### 助手

MTP 上如果要让远程仓库 `origin/main` 无条件覆盖本地代码，在启动脚本开头、正式训练前加入：

```bash
REPO_DIR=/opt/huawei/schedule-train/algorithm

git -C "${REPO_DIR}" fetch origin main
git -C "${REPO_DIR}" reset --hard origin/main
git -C "${REPO_DIR}" clean -fd

echo "Current commit:"
git -C "${REPO_DIR}" rev-parse HEAD
```

含义：

- `reset --hard origin/main`：覆盖所有已跟踪文件的本地修改。
- `clean -fd`：删除仓库内未跟踪的文件和目录。
- 不建议用 `git pull -X theirs`，它并不等于完整强制覆盖。

脚本可以改为：

```bash
ROOT_PATH=/opt/huawei
REPO_DIR="${ROOT_PATH}/schedule-train/algorithm"
PROJECT_ROOT=/cache/algorithm

git -C "${REPO_DIR}" fetch origin main
git -C "${REPO_DIR}" reset --hard origin/main
git -C "${REPO_DIR}" clean -fd

cp -r "${REPO_DIR}" "${PROJECT_ROOT}"
```

必须先把 HPT 修复提交并推送到 `origin/main`：

```bash
git add verl/experimental/hpt/algorithm.py \
        verl/experimental/hpt/tests/test_algorithm.py
git commit -m "fix(hpt): support Qwen3.5 text-only multi-axis position ids"
git push origin main
```

注意：`git clean -fd` 会删除 MTP 仓库目录中的所有未跟踪文件，训练数据和输出目录不要放在该仓库目录内。

---

## 3. 第二次报错：HPT 批次不能被 48 个 DP rank 整除

### 用户

再次报错：SFT-RL/HPT-verl090-verl/HPT-verl090-verl/wrong_txt/1014_2.txt

### 助手

新报错是下一层多节点兼容问题，上一处 Qwen3.5 修复已经生效。

当前 actor DP 数为 48，但 HPT 仍固定按 8 补齐批次：

```text
batch size = 640
chunks = 48
640 % 48 != 0
```

已修复为按照 `lcm(8, actor_dp_size)` 补齐。本次会将 640 临时补到 672，计算完成后移除 padding。

修改文件：

- `verl/experimental/hpt/layout.py`
- `verl/trainer/ppo/ray_trainer.py`
- `verl/experimental/hpt/tests/test_layout.py`

已通过语法检查、脚本检查和 `git diff --check`。本地缺少 PyTorch/pytest，完整运行仍需在 MTP 验证。

提交并推送：

```bash
cd SFT-RL/HPT-verl090-verl/HPT-verl090-verl

git add \
  verl/experimental/hpt/layout.py \
  verl/trainer/ppo/ray_trainer.py \
  verl/experimental/hpt/tests/test_layout.py

git commit -m "fix(hpt): pad routed batches for multi-node data parallelism"
git push origin main
```

然后让 MTP 强制同步 `origin/main` 后重新运行。仅调整 `TRAIN_BATCH_SIZE` 不能可靠解决，因为 HPT 路由后的批次大小会随每批成功率动态变化。

---

## 4. 第三次报错：多 worker MFU 指标返回 list

### 用户

SFT-RL/HPT-verl090-verl/HPT-verl090-verl/wrong_txt/1014_3.txt，再次报错，我希望你仔细检查

### 助手

仔细检查后确认，前两项修复均已生效，程序已经：

1. 完成 Qwen3.5 三维 `position_ids` 替换。
2. 将 640 条 HPT 数据补齐到 672 条并成功切分到 48 个 DP rank。
3. 完成 old-log-prob 模型前向。

当前报错发生在读取 worker 返回的 MFU 指标：

```text
TypeError: list indices must be integers or slices, not str
```

48 个 worker 汇总后，`metrics` 是 `list[dict]`，原代码却固定按单个 dict 访问：

```python
tu.get(output, "metrics")["mfu"]
```

现已改为兼容单 worker dict、多 worker list 以及嵌套数值，并取平均 MFU。

此外提前修复了两个确定会出现的后续问题：

- 支持脚本使用的 `ppo_micro_batch_size_per_gpu=1`，不再强制要求废弃的 `ppo_micro_batch_size`。当前配置计算结果为：

```text
global mini batch = 6 × 8 = 48
global micro batch = 1 × 48 = 48
gradient accumulation = 1
```

- 保留最终 padding 行参与各 rank 的同步前向/反向，但通过现有 `hpt_valid_mask` 将其 loss 置零。否则 32 条 padding 被各 rank 独立删除后，48 个 rank 的迭代次数不同，容易造成 HCCL/FSDP 卡死。

修改位置：

- `verl/trainer/ppo/ray_trainer.py`
- `verl/experimental/hpt/tests/test_trainer_helpers.py`

已通过：

- Python 语法检查
- Shell 脚本语法检查
- `git diff --check`
- 48-DP 尺寸推演：`640 + 32 = 672`，每个 rank 正好 14 条

本地没有 PyTorch/pytest，因此完整分布式测试仍需在 MTP 运行。

只提交本次两个文件，避免误提交工作区中已有的其他未跟踪文件：

```bash
cd SFT-RL/HPT-verl090-verl/HPT-verl090-verl

git add \
  verl/trainer/ppo/ray_trainer.py \
  verl/experimental/hpt/tests/test_trainer_helpers.py

git commit -m "fix(hpt): handle multi-worker metrics and padded actor updates"
git push origin main
```

MTP 强制同步 `origin/main` 后直接重跑原 HPT 脚本，不需要修改启动参数。

---

## 5. 导出请求

### 用户

将当前对话导出到SFT-RL/HPT-verl090-verl/HPT-verl090-verl/wrong_txt/目录下

