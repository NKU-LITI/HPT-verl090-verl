# Copyright 2026 The verl authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Score HPT Table-2 rollouts and build accuracy/token/truncation reports.

The ``score`` command consumes parquet files written by
``verl.trainer.main_generation_server``.  It explodes every sampled response
to JSONL, applies the repository's HPT verifier, and reports avg@N per prompt.
The ``aggregate`` command combines SFT/DFT/GRPO/HPT runs into one Table-2-style
CSV and Markdown table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

BENCHMARK_ORDER = ["AIME24", "AIME25", "AMC", "MATH-500", "Minerva", "Olympiad", "ARC-c", "GPQA"]
ID_BENCHMARKS = BENCHMARK_ORDER[:6]
OOD_BENCHMARKS = BENCHMARK_ORDER[6:]
STAGE_ORDER = ["Base", "SFT", "DFT", "GRPO", "HPT"]


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return str(value)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _as_list(value: Any) -> list:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [value]


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return int(value)


def infer_benchmark(data_source: Any, source_file: Any) -> str:
    """Map the common HPT directory/data-source spellings to paper columns."""
    key = f"{source_file or ''} {data_source or ''}".lower().replace("_", "-")
    aliases = [
        ("AIME24", ("aime24", "aime-24", "aime2024", "aime-2024")),
        ("AIME25", ("aime25", "aime-25", "aime2025", "aime-2025")),
        ("MATH-500", ("math-500", "math500")),
        ("Minerva", ("minerva",)),
        ("Olympiad", ("olympiad",)),
        ("ARC-c", ("arc-c", "arc_challenge", "arc-challenge", "arcchallenge")),
        ("GPQA", ("gpqa",)),
        ("AMC", ("amc",)),
    ]
    for canonical, names in aliases:
        if any(name in key for name in names):
            return canonical
    return str(data_source or "unknown")


def _ground_truth(row: pd.Series):
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict) or hasattr(reward_model, "get"):
        ground_truth = reward_model.get("ground_truth")
    else:
        ground_truth = row.get("ground_truth")
    # Arrow commonly materializes list-valued ground truths as ndarrays, while
    # the HPT verifier intentionally accepts Python lists of alternative
    # answers.  Normalize here instead of silently grading every alternative
    # answer as invalid.
    if isinstance(ground_truth, np.ndarray):
        return ground_truth.tolist()
    return ground_truth


def _aligned(values: Any, size: int, default=None) -> list:
    result = _as_list(values)
    if not result:
        return [default] * size
    if len(result) == 1 and size > 1:
        return result * size
    if len(result) != size:
        raise ValueError(f"sample metadata has {len(result)} values, expected {size}")
    return result


def load_generated_rollouts(paths: list[Path], stage: str, fallback_max_tokens: int) -> list[dict]:
    """Explode generated parquet rows into one record per sampled response."""
    rollouts: list[dict] = []
    for path in paths:
        frame = pd.read_parquet(path)
        if "responses" not in frame:
            raise KeyError(f"{path} has no 'responses' column")
        for row_number, row in frame.iterrows():
            responses = _as_list(row["responses"])
            if not responses:
                continue
            count = len(responses)
            response_counts = _aligned(row.get("response_token_counts"), count)
            prompt_counts = _aligned(row.get("prompt_token_counts"), count)
            total_counts = _aligned(row.get("total_token_counts"), count)
            finish_reasons = _aligned(row.get("finish_reasons"), count)
            source_file = row.get("__source_file__", str(path))
            data_source = row.get("data_source", "unknown")
            benchmark = infer_benchmark(data_source, source_file)
            max_tokens = _optional_int(row.get("generation_max_tokens")) or fallback_max_tokens
            raw_uid = row.get("uid")
            if raw_uid is None or (isinstance(raw_uid, float) and math.isnan(raw_uid)):
                raw_uid = row_number
            # Prefix with the file because AMC22 and AMC23 can both contain
            # local integer UIDs.  Treating those as one prompt would turn two
            # avg@32 problems into one incorrect avg@64 problem.
            base_uid = f"{source_file}#{raw_uid}"
            for sample_index, response in enumerate(responses):
                response_tokens = _optional_int(response_counts[sample_index])
                finish_reason = finish_reasons[sample_index]
                if finish_reason is not None:
                    truncated = str(finish_reason).lower() in {"length", "max_tokens"}
                else:
                    truncated = response_tokens is not None and response_tokens >= max_tokens
                rollouts.append(
                    {
                        "stage": stage,
                        "benchmark": benchmark,
                        "data_source": str(data_source),
                        "source_file": str(source_file),
                        "prompt_uid": str(base_uid),
                        "sample_index": sample_index,
                        "prompt": row.get("prompt"),
                        "response": str(response or ""),
                        "ground_truth": _ground_truth(row),
                        "prompt_tokens": _optional_int(prompt_counts[sample_index]),
                        "response_tokens": response_tokens,
                        "total_tokens": _optional_int(total_counts[sample_index]),
                        "max_response_tokens": max_tokens,
                        "finish_reason": None if finish_reason is None else str(finish_reason),
                        "truncated": bool(truncated),
                        "temperature": row.get("generation_temperature"),
                        "top_p": row.get("generation_top_p"),
                    }
                )
    return rollouts


def _score_one(payload: tuple[str, str, Any, int]) -> tuple[float, str | None, bool]:
    data_source, response, ground_truth, max_response_tokens = payload
    try:
        from verl.experimental.hpt.reward import (
            compute_score,
            extract_answer,
            final_response,
        )

        final = final_response(response)
        format_valid = final is not None and extract_answer(final) is not None
        score = compute_score(
            data_source=data_source,
            solution_str=response,
            ground_truth=ground_truth,
            extra_info={"response_length": None},
            max_response_length=max_response_tokens,
        )
        return float(score), None, format_valid
    except Exception as exc:  # noqa: BLE001 - one bad symbolic expression must not abort an evaluation.
        return 0.0, f"{type(exc).__name__}: {exc}", False


def _apply_verdicts(rollouts: list[dict], verdicts) -> None:
    stage = rollouts[0].get("stage", "") if rollouts else ""
    description = f"Scoring {stage} rollouts" if stage else "Scoring rollouts"
    # MTP captures stderr rather than providing a TTY, so keep tqdm enabled.
    # Throttling refreshes prevents one remote-log update per scored sample.
    with tqdm(
        verdicts,
        total=len(rollouts),
        desc=description,
        unit="sample",
        mininterval=2.0,
        dynamic_ncols=True,
        disable=False,
    ) as progress:
        for row, (score, error, format_valid) in zip(rollouts, progress, strict=True):
            row.update(score=score, score_error=error, format_valid=format_valid)


def score_rollouts(rollouts: list[dict], workers: int) -> None:
    payloads = [
        (row["data_source"], row["response"], row["ground_truth"], row["max_response_tokens"])
        for row in rollouts
    ]
    if workers == 1:
        _apply_verdicts(rollouts, map(_score_one, payloads))
        return
    with Pool(processes=workers, maxtasksperchild=256) as pool:
        verdicts = pool.imap(_score_one, payloads, chunksize=16)
        _apply_verdicts(rollouts, verdicts)


def _mode(values: list[int]) -> int:
    return Counter(values).most_common(1)[0][0]


def _finite_numbers(rows: list[dict], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]


def summarize_benchmark(stage: str, benchmark: str, rows: list[dict]) -> dict:
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_prompt[row["prompt_uid"]].append(row)
    samples_per_prompt = [len(items) for items in by_prompt.values()]
    prompt_scores = [float(np.mean([item["score"] for item in items])) for items in by_prompt.values()]
    response_tokens = _finite_numbers(rows, "response_tokens")
    prompt_tokens = _finite_numbers(rows, "prompt_tokens")
    total_tokens = _finite_numbers(rows, "total_tokens")
    correct_tokens = _finite_numbers([row for row in rows if row["score"] > 0], "response_tokens")
    incorrect_tokens = _finite_numbers([row for row in rows if row["score"] <= 0], "response_tokens")

    def mean_or_none(values):
        return float(np.mean(values)) if values else None

    def percentile_or_none(values, percentile):
        return float(np.percentile(values, percentile)) if values else None

    truncated_count = sum(bool(row["truncated"]) for row in rows)
    return {
        "stage": stage,
        "benchmark": benchmark,
        "problems": len(by_prompt),
        "samples": len(rows),
        "samples_per_problem": _mode(samples_per_prompt),
        "min_samples_per_problem": min(samples_per_prompt),
        "max_samples_per_problem": max(samples_per_prompt),
        "avg_at_n": float(np.mean(prompt_scores)),
        "avg_at_n_percent": 100.0 * float(np.mean(prompt_scores)),
        "prompt_level_std": float(np.std(prompt_scores)),
        "correct_count": sum(row["score"] > 0 for row in rows),
        "format_valid_count": sum(bool(row["format_valid"]) for row in rows),
        "format_valid_rate": float(np.mean([row["format_valid"] for row in rows])),
        "score_errors": sum(row["score_error"] is not None for row in rows),
        "prompt_token_count": len(prompt_tokens),
        "prompt_tokens_total": int(sum(prompt_tokens)) if prompt_tokens else None,
        "prompt_tokens_mean": mean_or_none(prompt_tokens),
        "response_token_count": len(response_tokens),
        "response_tokens_total": int(sum(response_tokens)) if response_tokens else None,
        "response_tokens_mean": mean_or_none(response_tokens),
        "response_tokens_median": percentile_or_none(response_tokens, 50),
        "response_tokens_p90": percentile_or_none(response_tokens, 90),
        "response_tokens_p95": percentile_or_none(response_tokens, 95),
        "response_tokens_max": int(max(response_tokens)) if response_tokens else None,
        "correct_response_tokens_mean": mean_or_none(correct_tokens),
        "incorrect_response_tokens_mean": mean_or_none(incorrect_tokens),
        "total_tokens_total": int(sum(total_tokens)) if total_tokens else None,
        "truncated_count": truncated_count,
        "truncation_rate": truncated_count / len(rows),
    }


def summarize(stage: str, rollouts: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rollouts:
        grouped[row["benchmark"]].append(row)
    ordered = [name for name in BENCHMARK_ORDER if name in grouped]
    ordered.extend(sorted(set(grouped) - set(ordered)))
    return [summarize_benchmark(stage, benchmark, grouped[benchmark]) for benchmark in ordered]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty summary")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def run_score(args) -> None:
    rollouts = load_generated_rollouts(args.input, args.stage, args.max_response_tokens)
    if not rollouts:
        raise ValueError("no responses found in the supplied parquet files")
    score_rollouts(rollouts, args.workers)
    summary = summarize(args.stage, rollouts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl = "".join(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n" for row in rollouts)
    _atomic_text(args.output_dir / "rollouts.jsonl", jsonl)
    _write_csv(args.output_dir / "benchmark_summary.csv", summary)
    _atomic_text(
        args.output_dir / "summary.json",
        json.dumps({"stage": args.stage, "benchmarks": summary}, ensure_ascii=False, indent=2, default=_json_default)
        + "\n",
    )
    print(f"[{args.stage}] wrote {len(rollouts)} rollouts to {args.output_dir}")
    for row in summary:
        print(
            f"  {row['benchmark']}: avg@{row['samples_per_problem']}={row['avg_at_n_percent']:.2f} "
            f"tokens={row['response_tokens_mean']!s} truncation={100 * row['truncation_rate']:.2f}%"
        )


def _stage_sort_key(stage: str):
    try:
        return STAGE_ORDER.index(stage), stage
    except ValueError:
        return len(STAGE_ORDER), stage


def _mean_available(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else None


def run_aggregate(args) -> None:
    summaries: list[dict] = []
    for run_dir in args.run_dir:
        path = run_dir / "benchmark_summary.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        summaries.extend(pd.read_csv(path).replace({np.nan: None}).to_dict(orient="records"))
    by_stage: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in summaries:
        by_stage[str(row["stage"])][str(row["benchmark"])] = row

    table_rows = []
    token_rows = []
    for stage in sorted(by_stage, key=_stage_sort_key):
        bench = by_stage[stage]
        table_row: dict[str, Any] = {"stage": stage}
        for name in BENCHMARK_ORDER:
            table_row[name] = bench.get(name, {}).get("avg_at_n_percent")
        table_row["ID Avg"] = _mean_available([table_row[name] for name in ID_BENCHMARKS])
        table_row["OOD Avg"] = _mean_available([table_row[name] for name in OOD_BENCHMARKS])
        table_rows.append(table_row)

        samples = sum(int(row["samples"]) for row in bench.values())
        response_count = sum(int(row.get("response_token_count") or 0) for row in bench.values())
        prompt_count = sum(int(row.get("prompt_token_count") or 0) for row in bench.values())
        response_total = sum(int(row.get("response_tokens_total") or 0) for row in bench.values())
        prompt_total = sum(int(row.get("prompt_tokens_total") or 0) for row in bench.values())
        truncated = sum(int(row.get("truncated_count") or 0) for row in bench.values())
        format_valid = sum(int(row.get("format_valid_count") or 0) for row in bench.values())
        token_rows.append(
            {
                "stage": stage,
                "samples": samples,
                "token_usage_missing_samples": samples - response_count,
                "prompt_tokens_total": prompt_total,
                "response_tokens_total": response_total,
                "all_tokens_total": prompt_total + response_total,
                "prompt_tokens_mean": prompt_total / prompt_count if prompt_count else None,
                "response_tokens_mean": response_total / response_count if response_count else None,
                "format_valid_rate": format_valid / samples if samples else None,
                "truncated_count": truncated,
                "truncation_rate": truncated / samples if samples else None,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "table2.csv", table_rows)
    _write_csv(args.output_dir / "token_and_truncation.csv", token_rows)
    _write_csv(args.output_dir / "all_benchmark_summary.csv", summaries)

    columns = ["stage", *ID_BENCHMARKS, "ID Avg", *OOD_BENCHMARKS, "OOD Avg"]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |"
    lines = [header, separator]
    for row in table_rows:
        cells = [row["stage"]]
        for column in columns[1:]:
            value = row.get(column)
            cells.append("--" if value is None else f"{float(value):.1f}")
        lines.append("| " + " | ".join(cells) + " |")
    _atomic_text(args.output_dir / "table2.md", "\n".join(lines) + "\n")
    print(f"wrote combined Table-2 report to {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="score generated parquet files")
    score_parser.add_argument("--input", type=Path, action="append", required=True, help="repeat for each parquet")
    score_parser.add_argument("--stage", required=True, help="Base/SFT/DFT/GRPO/HPT or another checkpoint label")
    score_parser.add_argument("--output-dir", type=Path, required=True)
    score_parser.add_argument("--workers", type=int, default=8)
    score_parser.add_argument("--max-response-tokens", type=int, default=8192)
    score_parser.set_defaults(func=run_score)

    aggregate_parser = subparsers.add_parser("aggregate", help="combine stage-level reports")
    aggregate_parser.add_argument("--run-dir", type=Path, action="append", required=True)
    aggregate_parser.add_argument("--output-dir", type=Path, required=True)
    aggregate_parser.set_defaults(func=run_aggregate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "workers", 1) < 1:
        raise ValueError("--workers must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
