#!/usr/bin/env python3
"""Prepare the ARC-Challenge test split for the MTP HPT evaluator."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd

DATASET_NAME = "allenai/ai2_arc"
DATASET_CONFIG = "ARC-Challenge"
# Hugging Face commit that converted the official dataset to parquet.
DEFAULT_REVISION = "16644173ab25fa007c815e3204a8b59a4034ba1b"
EXPECTED_TEST_ROWS = 1172
OPTION_RE = re.compile(r"(?m)^(\s*)([A-Z0-9]+)[.)]\s+")


def _choice_columns(choices: Any) -> tuple[list[str], list[str]]:
    if not isinstance(choices, dict):
        raise TypeError(f"choices must be a mapping, got {type(choices).__name__}")
    labels = [str(value).strip() for value in choices.get("label", [])]
    texts = [str(value).strip() for value in choices.get("text", [])]
    if not 3 <= len(labels) <= 5 or len(labels) != len(texts):
        raise ValueError(f"invalid ARC choices: labels={labels!r}, texts={texts!r}")
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate ARC labels: {labels!r}")
    return labels, texts


def convert_example(example: dict[str, Any], index: int) -> dict[str, Any]:
    """Convert one Hugging Face ARC example to the local HPT parquet schema."""
    old_labels, texts = _choice_columns(example["choices"])
    new_labels = [chr(ord("A") + offset) for offset in range(len(old_labels))]
    label_map = dict(zip(old_labels, new_labels, strict=True))
    answer_key = str(example["answerKey"]).strip()
    if answer_key not in label_map:
        raise ValueError(f"answer key {answer_key!r} is not among {old_labels!r}")

    options = "\n".join(f"({label}) {text}" for label, text in zip(new_labels, texts, strict=True))
    boxed_choices = ", ".join(f"\\boxed{{{label}}}" for label in new_labels[:-1])
    boxed_choices += f", or \\boxed{{{new_labels[-1]}}}"
    question = str(example["question"]).strip()
    prompt = f"{question}\n\n{options}\n\nPlease write your final answer in the form of {boxed_choices}."

    example_id = str(example.get("id") or index)
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "source": DATASET_NAME,
        "id": example_id,
        "data_source": "ARC-c",
        "ability": "science",
        "reward_model": {"style": "rule", "ground_truth": label_map[answer_key]},
        "extra_info": {"split": "test", "index": f"ARC-c-{index}", "source_id": example_id},
    }


def convert_local_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    """Normalize a local ARC parquet row to the DeepMath choice prompt."""
    messages = [dict(message) for message in list(row["prompt"])]
    user_messages = [message for message in messages if message.get("role") == "user"]
    if len(user_messages) != 1:
        raise ValueError(f"row {index}: expected exactly one user message, got {len(user_messages)}")

    question = str(user_messages[0]["content"])
    matches = list(OPTION_RE.finditer(question))
    old_labels = [match.group(2) for match in matches]
    if not 3 <= len(old_labels) <= 5 or len(old_labels) != len(set(old_labels)):
        raise ValueError(f"row {index}: unexpected ARC choices {old_labels!r}")

    reward_model = dict(row["reward_model"])
    truth = str(reward_model["ground_truth"]).strip()
    if truth not in old_labels:
        raise ValueError(f"row {index}: ground truth {truth!r} is not among {old_labels!r}")

    new_labels = [chr(ord("A") + offset) for offset in range(len(old_labels))]
    label_map = dict(zip(old_labels, new_labels, strict=True))

    def replace_label(match: re.Match[str]) -> str:
        return f"{match.group(1)}({label_map[match.group(2)]}) "

    question = OPTION_RE.sub(replace_label, question).rstrip()
    boxed_choices = ", ".join(f"\\boxed{{{label}}}" for label in new_labels[:-1])
    boxed_choices += f", or \\boxed{{{new_labels[-1]}}}"
    prompt = f"{question}\n\nPlease write your final answer in the form of {boxed_choices}."

    extra_info = dict(row.get("extra_info") or {})
    extra_info.update(split="test", index=f"ARC-c-{index}")
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "data_source": "ARC-c",
        "ability": row.get("ability", "science"),
        "reward_model": {"style": "rule", "ground_truth": label_map[truth]},
        "extra_info": extra_info,
    }


def _write_and_validate(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)

    # Fail during preparation, rather than after an expensive rollout, if the
    # nested parquet schema was not serialized as expected.
    check = pd.read_parquet(output)
    required = {"prompt", "data_source", "reward_model", "extra_info"}
    missing = required - set(check.columns)
    if len(check) != EXPECTED_TEST_ROWS or missing:
        raise RuntimeError(f"invalid output {output}: rows={len(check)}, missing_columns={sorted(missing)}")
    print(f"Wrote {len(check)} ARC-Challenge test rows to {output}")


def prepare(output: Path, revision: str) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: install it with `python3 -m pip install datasets`.") from exc

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test", revision=revision)
    if len(dataset) != EXPECTED_TEST_ROWS:
        raise ValueError(f"expected {EXPECTED_TEST_ROWS} ARC-Challenge test rows, got {len(dataset)}")

    _write_and_validate([convert_example(dict(example), index) for index, example in enumerate(dataset)], output)


def prepare_from_parquet(source: Path, output: Path) -> None:
    frame = pd.read_parquet(source)
    if len(frame) != EXPECTED_TEST_ROWS:
        raise ValueError(f"expected {EXPECTED_TEST_ROWS} ARC-Challenge test rows, got {len(frame)}")
    rows = [convert_local_row(row, index) for index, row in enumerate(frame.to_dict(orient="records"))]
    _write_and_validate(rows, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path, normally $zlt_data/HPT_data/ARC-c/test.parquet",
    )
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Pinned allenai/ai2_arc revision")
    parser.add_argument(
        "--source-parquet",
        type=Path,
        help="Normalize an existing 1,172-row ARC-Challenge parquet instead of downloading it",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.source_parquet is None:
        prepare(args.output, args.revision)
    else:
        prepare_from_parquet(args.source_parquet, args.output)
