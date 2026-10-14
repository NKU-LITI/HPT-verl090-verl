#!/usr/bin/env python3
"""
监控训练日志文件，每 N 分钟更新一次训练曲线图。

用法:
    # 持续监控（每 600 秒更新一次）
    python3 plot_curves.py /path/to/train.log /path/to/plots/

    # 只生成一次（训练结束后调用）
    python3 plot_curves.py /path/to/train.log /path/to/plots/ --once
"""

import argparse
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 不需要 GUI
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# 需要画图的指标分组，key=图表名, value=日志中的 key 列表
METRIC_GROUPS = {
    "loss": [
        "actor/loss",
        "actor/pg_loss",
        "actor/kl_loss",
        "distillation/distill_loss",
    ],
    "reward": [
        "reward/mean",
        "reward/std",
        "reward/min",
        "reward/max",
    ],
    "gradient": [
        "actor/grad_norm",
        "actor/lr",
    ],
    "response": [
        "response_length/mean",
        "prompt_length/mean",
    ],
    "timing": [
        "timing/gen",
        "timing/reward",
        "timing/old_log_prob",
        "timing/ref_log_prob",
        "timing/update_actor",
    ],
}

# 中文字体回退
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


# 匹配 Ray 日志前缀，如 "(TaskRunnerV1 pid=143192) "
RAY_PREFIX_RE = re.compile(r"^\([^)]+\)\s+")

# ANSI 转义序列 (Ray 日志带颜色代码)
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def parse_log_line(line: str) -> dict | None:
    """解析一行日志，返回 {step: N, "actor/loss": 0.123, ...} 或 None"""
    # 去掉 ANSI 颜色代码和 Ray actor 前缀
    line = ANSI_ESCAPE_RE.sub("", line)
    line = RAY_PREFIX_RE.sub("", line).strip()
    if not line.startswith("step:"):
        return None

    parts = line.strip().split(" - ")
    if len(parts) < 2:
        return None

    result = {}
    # 第一段: "step:0"
    step_match = re.match(r"step:(\d+)", parts[0])
    if not step_match:
        return None
    result["_step"] = int(step_match.group(1))

    # 后续每段: "key:value"
    for part in parts[1:]:
        kv = part.split(":", 1)
        if len(kv) != 2:
            continue
        key, val = kv[0], kv[1]
        try:
            result[key] = float(val)
        except ValueError:
            continue

    return result


def parse_jsonl_file(log_path: str) -> dict[str, list[tuple[int, float]]]:
    """解析 jsonl 格式的指标文件，返回 {metric_key: [(step, value), ...]}"""
    import json

    data: dict[str, list[tuple[int, float]]] = defaultdict(list)

    if not os.path.exists(log_path):
        return dict(data)

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = record.get("step", 0)
            metrics = record.get("data", {})
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    data[key].append((step, float(val)))
                elif isinstance(val, list):
                    # handle list of numbers (e.g., histogram data)
                    pass

    return dict(data)


def parse_log_file(log_path: str) -> dict[str, list[tuple[int, float]]]:
    """解析日志文件（自动检测格式），返回 {metric_key: [(step, value), ...]}"""
    if not os.path.exists(log_path):
        return {}

    # 根据扩展名或内容自动选择解析方式
    if log_path.endswith(".jsonl"):
        return parse_jsonl_file(log_path)

    # 否则尝试 jsonl，如果第一条是 json 就用 jsonl 模式
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline().strip()
    if first.startswith("{") and first.endswith("}"):
        return parse_jsonl_file(log_path)

    # 旧格式: "step:N - key:val - ..."
    data: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = parse_log_line(line)
            if parsed is None:
                continue
            step = parsed.pop("_step")
            for key, val in parsed.items():
                data[key].append((step, val))
    return dict(data)


def make_plot(ax, steps, values, label, color):
    """在 ax 上画一条线"""
    if len(steps) < 1:
        return
    ax.plot(steps, values, linewidth=0.8, color=color, label=label, alpha=0.85)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))


def generate_plots(data: dict, output_dir: str):
    """根据解析到的数据生成所有图表"""
    os.makedirs(output_dir, exist_ok=True)

    if not data:
        print("[plot_curves] No training data found in log yet.")
        return

    total_steps = max((s for vals in data.values() for s, _ in vals), default=0)
    print(f"[plot_curves] Parsed {len(data)} metrics across {total_steps} steps.")

    # 色板
    colors = plt.cm.tab10.colors

    for group_name, metric_keys in METRIC_GROUPS.items():
        fig, ax = plt.subplots(figsize=(10, 4))

        has_data = False
        for i, key in enumerate(metric_keys):
            if key in data and data[key]:
                has_data = True
                steps, vals = zip(*data[key])
                color = colors[i % len(colors)]
                make_plot(ax, steps, vals, label=key.split("/")[-1], color=color)

        if not has_data:
            plt.close(fig)
            continue

        ax.set_xlabel("Step")
        ax.set_title(f"Training: {group_name}")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{group_name}.png"), dpi=120)
        plt.close(fig)
        print(f"[plot_curves] Updated {group_name}.png")

    # ---- 汇总大图 ----
    n_groups = sum(1 for keys in METRIC_GROUPS.values() if any(k in data and data[k] for k in keys))
    if n_groups == 0:
        return

    fig, axes = plt.subplots(n_groups, 1, figsize=(12, 3.5 * n_groups), sharex=True)
    if n_groups == 1:
        axes = [axes]

    idx = 0
    for group_name, metric_keys in METRIC_GROUPS.items():
        if not any(k in data and data[k] for k in metric_keys):
            continue
        ax = axes[idx]
        for i, key in enumerate(metric_keys):
            if key in data and data[key]:
                steps, vals = zip(*data[key])
                color = colors[i % len(colors)]
                ax.plot(steps, vals, linewidth=0.8, color=color, label=key, alpha=0.85)
        ax.legend(fontsize=6, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(group_name)
        idx += 1

    axes[-1].set_xlabel("Step")
    fig.suptitle(f"Training Curves (up to step {total_steps})", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "all_metrics.png"), dpi=120)
    plt.close(fig)
    print(f"[plot_curves] Updated all_metrics.png (total {total_steps} steps)")


def main():
    parser = argparse.ArgumentParser(description="Monitor training log and plot curves")
    parser.add_argument("log_file", help="Path to the training log file")
    parser.add_argument("output_dir", help="Directory to save plot images")
    parser.add_argument(
        "--interval", type=int, default=600,
        help="Seconds between each update (default: 600 = 10 min)"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Generate plots once and exit (no monitoring)"
    )
    args = parser.parse_args()

    print(f"[plot_curves] Log file: {args.log_file}")
    print(f"[plot_curves] Output  : {args.output_dir}")
    if args.once:
        print("[plot_curves] Mode: once")
    else:
        print(f"[plot_curves] Mode: watch, interval={args.interval}s")

    while True:
        data = parse_log_file(args.log_file)
        generate_plots(data, args.output_dir)

        if args.once:
            break

        # 检查训练是否结束（父进程已退出）
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
