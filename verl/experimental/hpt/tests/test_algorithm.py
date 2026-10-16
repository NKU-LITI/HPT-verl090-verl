from types import SimpleNamespace

import torch

from verl.experimental.hpt.algorithm import _replace_with_target
from verl.trainer.ppo.grpo_metrics import compute_grpo_rollout_metrics


class _Tokenizer:
    eos_token_id = 11
    pad_token_id = 0

    def __call__(self, text, *, add_special_tokens, return_attention_mask):
        assert text == "target"
        assert not add_special_tokens
        assert not return_attention_mask
        return {"input_ids": [9, 10]}


def test_replace_target_supports_text_only_multi_axis_position_ids():
    attention_mask = torch.tensor([[0, 1, 1, 1, 1, 0, 0]])
    position_ids = torch.tensor([[[0, 0, 1, 2, 3, 3, 3]]]).expand(-1, 4, -1).clone()
    batch = SimpleNamespace(
        batch={
            "prompts": torch.tensor([[0, 5, 6]]),
            "responses": torch.tensor([[7, 8, 0, 0]]),
            "response_mask": torch.tensor([[1, 1, 0, 0]]),
            "input_ids": torch.tensor([[0, 5, 6, 7, 8, 0, 0]]),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        non_tensor_batch={"hpt_target": ["target"]},
    )

    _replace_with_target(batch, 0, _Tokenizer())

    expected = torch.tensor([0, 0, 1, 2, 3, 4, 4])
    assert torch.equal(batch.batch["responses"][0], torch.tensor([9, 10, 11, 0]))
    assert torch.equal(batch.batch["position_ids"][0], expected.unsqueeze(0).expand(4, -1))


def test_compute_grpo_rollout_metrics_emits_batch_and_on_policy_metrics():
    batch = SimpleNamespace(
        batch={
            "responses": torch.tensor(
                [
                    [1, 2, 0],
                    [1, 2, 0],
                    [3, 4, 5],
                    [6, 0, 0],
                ]
            ),
            "response_mask": torch.tensor(
                [
                    [1, 1, 0],
                    [1, 1, 0],
                    [1, 1, 1],
                    [1, 0, 0],
                ]
            ),
        },
        non_tensor_batch={"uid": torch.tensor([10, 10, 20, 20]).numpy()},
    )
    reward_tensor = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ]
    )

    metrics = compute_grpo_rollout_metrics(batch, reward_tensor)

    assert metrics["batch/solve_none"] == 0.0
    assert metrics["batch/solve_none_format"] == 0.0
    assert metrics["batch/solve_all"] == 0.0
    assert metrics["batch/solved"] == 0.5
    assert metrics["batch/failed"] == 0.25
    assert metrics["batch/success_count_0"] == 0.0
    assert metrics["batch/success_count_1"] == 2.0
    assert metrics["batch/success_count_2"] == 0.0
    assert metrics["batch/success_count_mean"] == 1.0
    assert metrics["batch/success_count_std"] == 0.0
    assert metrics["batch/unique_responses_per_group"] == 1.5
    assert metrics["batch/unique_response_ratio"] == 0.75
    assert metrics["batch/duplicate_response_groups"] == 1.0
    assert metrics["on_off_metrics/on_response_length_mean"] == 2.0
    assert metrics["on_off_metrics/on_score"] == 0.25
    assert metrics["on_off_metrics/off_example_ratio"] == 0.0
    assert torch.isnan(torch.tensor(metrics["on_off_metrics/off_response_length_mean"]))
    assert torch.isnan(torch.tensor(metrics["on_off_metrics/off_score"]))


def test_compute_grpo_rollout_metrics_honors_reward_overrides():
    batch = SimpleNamespace(
        batch={
            "responses": torch.tensor([[1], [2]]),
            "response_mask": torch.ones(2, 1, dtype=torch.long),
        },
        non_tensor_batch={"uid": torch.tensor([1, 1]).numpy()},
    )
    reward_tensor = torch.tensor([[2.0], [-2.0]])

    metrics = compute_grpo_rollout_metrics(
        batch,
        reward_tensor,
        config={"success_reward": 2.0, "failure_reward": -2.0},
    )

    assert metrics["batch/solved"] == 0.5
    assert metrics["batch/failed"] == 0.5
    assert metrics["batch/success_count_1"] == 1.0
