import numpy as np

from verl.experimental.hpt.table2_eval import infer_benchmark, summarize_benchmark


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
