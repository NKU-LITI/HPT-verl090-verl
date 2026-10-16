# HPT Table 2 evaluation on MTP

Chinese model-export and evaluation guide: [README_zh.md](README_zh.md).

This directory evaluates the SFT, DFT, and GRPO checkpoints with the protocol
from HPT Table 2.  DFT is an additional comparison row; it is not a row in the
paper's original Table 2.

## Protocol

- AIME24, AIME25, AMC: temperature 0.6, top-p 0.95, avg@32.
- MATH-500, Minerva, Olympiad, ARC-c, GPQA: temperature 0.6, top-p 0.95, avg@1.
- Maximum response length: 8192 tokens.
- The verifier is `verl.experimental.hpt.reward.compute_score`, the same rule
  implementation used by the HPT/GRPO training scripts in this repository.

The paper defines AMC as AMC12 2022 + AMC12 2023.  The MTP training script only
references `AMC23/test.parquet`, so that is the default here.  To reproduce the
paper's combined benchmark, set:

```bash
AMC_FILES="${DATA_DIR}/AMC22/test.parquet,${DATA_DIR}/AMC23/test.parquet"
```

The data bundled with `HPT-verl090/data` uses the directory name
`Olympiad-Bench`, which is also the default here. This repository additionally
contains the missing 1,172-row ARC-Challenge test set at
`data/ARC-c/test.parquet`. To regenerate it from the pinned `allenai/ai2_arc`
source:

```bash
python3 examples/hpt_eval/prepare_arc_c.py \
  --output "${zlt_data}/HPT_data/ARC-c/test.parquet"
```

## 1. Merge checkpoints

SFT/DFT checkpoints point directly at `global_step_*`.  A GRPO checkpoint may
point at either `global_step_*` or `global_step_*/actor`.

```bash
SFT_CKPT=/path/to/sft/checkpoint/global_step_N \
DFT_CKPT=/path/to/dft/checkpoint/global_step_N \
GRPO_CKPT=/path/to/grpo/checkpoint/global_step_N \
bash examples/hpt_eval/merge_checkpoints.sh
```

## 2. Run on MTP

```bash
SFT_MODEL_PATH="${zlt_data}/eval_models/qwen35b_sft_hf" \
DFT_MODEL_PATH="${zlt_data}/eval_models/qwen35b_dft_hf" \
GRPO_MODEL_PATH="${zlt_data}/eval_models/qwen35b_grpo_hf" \
OUTPUT_ROOT="${zlt_data}/runs/eval_result/qwen35b_table2" \
bash examples/hpt_eval/run_table2_mtp.sh
```

All dataset locations have defaults under `${zlt_data}/HPT_data`, matching
`zlt_run_grpo_qwen35b_v3.sh`.  Override `AIME24_FILE`, `AIME25_FILE`,
`AMC_FILES`, `MATH500_FILE`, `MINERVA_FILE`, `OLYMPIAD_FILE`, `ARCC_FILE`, or
`GPQA_FILE` if the actual directory spelling differs. Missing verifier/runtime
packages are installed from the same Tsinghua PyPI mirror used by the training
script; override `PYPI_INDEX_URL` when needed.

## Outputs

Each stage directory contains:

- `generated/avg32.json` and `generated/avg1.json`: raw server outputs,
  including exact prompt/response token counts and finish reasons.
- `rollouts.jsonl`: one durable record per rollout, with answer, reward, token
  counts, format validity, and truncation flag.
- `benchmark_summary.csv`: accuracy, token distribution, format rate, and
  truncation rate by benchmark.

The final `report/` directory contains:

- `table2.md` and `table2.csv`: stage-by-benchmark results.
- `token_and_truncation.csv`: total/mean token counts and truncation rates by
  SFT/DFT/GRPO stage.
- `all_benchmark_summary.csv`: all detailed benchmark rows.

Existing generation JSON files are reused by default. Set
`RESUME_GENERATION=0` to sample them again.
