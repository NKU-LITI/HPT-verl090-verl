import json

import numpy as np

from verl.experimental.hpt import table2_eval
from verl.experimental.hpt.table2_eval import infer_benchmark, load_generated_rollouts, score_rollouts, summarize_benchmark


def _row(uid, score, tokens, truncated=False):
    return {
        "prompt_uid": uid,
        "score": score,
        "format_valid": True,
        "score_error": None,
        "prompt_tokens": 10,
        "response_tokens": tokens,
        "total_tokens": 10 + tokens,
        "truncated": truncated,
    }


def test_infer_benchmark_prefers_source_path():
    assert infer_benchmark("math", "/data/HPT_data/AIME25/test.parquet") == "AIME25"
    assert infer_benchmark("math", "/data/HPT_data/MATH-500/test.parquet") == "MATH-500"
    assert infer_benchmark("arc_challenge", "/unknown/test.parquet") == "ARC-c"


def test_load_generated_json_preserves_samples_and_metadata(tmp_path):
    path = tmp_path / "avg32.json"
    path.write_text(
        json.dumps(
            [
                {
                    "uid": "problem-1",
                    "__source_file__": "/data/AIME25/test.parquet",
                    "data_source": "math",
                    "prompt": [{"role": "user", "content": "1 + 1?"}],
                    "reward_model": {"ground_truth": "2"},
                    "responses": ["2", "3"],
                    "response_token_counts": [12, 32],
                    "finish_reasons": ["stop", "length"],
                    "generation_max_tokens": 32,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rows = load_generated_rollouts([path], "SFT", fallback_max_tokens=8192)

    assert [row["response"] for row in rows] == ["2", "3"]
    assert [row["sample_index"] for row in rows] == [0, 1]
    assert [row["truncated"] for row in rows] == [False, True]
    assert all(row["benchmark"] == "AIME25" and row["ground_truth"] == "2" for row in rows)


def test_summary_uses_prompt_level_avg_at_n_and_token_counts():
    rows = [
        _row("p1", 1.0, 100),
        _row("p1", 0.0, 200, truncated=True),
        _row("p2", 1.0, 300),
        _row("p2", 1.0, 400),
    ]
    summary = summarize_benchmark("SFT", "AIME24", rows)
    assert summary["samples_per_problem"] == 2
    assert summary["avg_at_n"] == 0.75
    assert summary["response_tokens_total"] == 1000
    assert summary["response_token_count"] == 4
    assert summary["response_tokens_mean"] == 250
    assert summary["truncation_rate"] == 0.25
    assert np.isclose(summary["prompt_level_std"], 0.25)


def test_score_rollouts_reports_progress(monkeypatch, capsys):
    rollouts = [
        {
            "stage": "SFT",
            "data_source": "math",
            "response": "first",
            "ground_truth": "1",
            "max_response_tokens": 32,
        },
        {
            "stage": "SFT",
            "data_source": "math",
            "response": "second",
            "ground_truth": "2",
            "max_response_tokens": 32,
        },
    ]
    monkeypatch.setattr(
        table2_eval,
        "_score_one",
        lambda payload: (float(payload[1] == "first"), None, True),
    )

    score_rollouts(rollouts, workers=1)

    progress_output = capsys.readouterr().err
    assert "Scoring SFT rollouts" in progress_output
    assert "2/2" in progress_output
    assert [row["score"] for row in rollouts] == [1.0, 0.0]
    assert [row["is_correct"] for row in rollouts] == [True, False]
