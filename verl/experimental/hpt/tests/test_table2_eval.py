import numpy as np

from verl.experimental.hpt import table2_eval
from verl.experimental.hpt.table2_eval import infer_benchmark, score_rollouts, summarize_benchmark


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
    monkeypatch.setattr(table2_eval, "_score_one", lambda _payload: (1.0, None, True))

    score_rollouts(rollouts, workers=1)

    progress_output = capsys.readouterr().err
    assert "Scoring SFT rollouts" in progress_output
    assert "2/2" in progress_output
    assert all(row["score"] == 1.0 for row in rollouts)
