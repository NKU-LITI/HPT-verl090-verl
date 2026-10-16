# Copyright 2026 The verl authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rollout-level metrics shared by pure GRPO and HPT training."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from verl import DataProto


def _get(config: DictConfig | dict, key: str, default: Any) -> Any:
    return config.get(key, default)


def _reward_matches(scores: torch.Tensor, value: float) -> torch.Tensor:
    """Match sequence rewards using the same tolerance as HPT routing."""
    return torch.isclose(scores, scores.new_tensor(value))


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    """Return a row-wise mean, or NaN when the selected side is empty."""
    selected = values[mask.to(device=values.device)]
    if selected.numel() == 0:
        return float("nan")
    return selected.float().mean().detach().item()


def compute_grpo_batch_metrics(
    scores: torch.Tensor,
    uids: np.ndarray,
    *,
    success_reward: float,
    failure_reward: float,
    format_reward: float,
) -> dict[str, float]:
    """Compute group outcome statistics for a GRPO rollout batch."""
    solve_none = 0
    solve_none_format = 0
    solve_all = 0
    success_counts = []
    for uid in np.unique(uids):
        group_scores = scores[np.flatnonzero(uids == uid).tolist()]
        success_counts.append(
            int(_reward_matches(group_scores, success_reward).sum().item())
        )
        if _reward_matches(group_scores, failure_reward).all():
            solve_none += 1
        elif _reward_matches(group_scores, success_reward).all():
            solve_all += 1
        elif _reward_matches(group_scores, format_reward).all():
            solve_none_format += 1

    metrics = {
        "batch/solve_none": float(solve_none),
        "batch/solve_none_format": float(solve_none_format),
        "batch/solve_all": float(solve_all),
        "batch/solved": _reward_matches(scores, success_reward).float().mean().item(),
        "batch/failed": _reward_matches(scores, failure_reward).float().mean().item(),
    }
    # HPT routes on the group-level best-of-n result, so retain the complete
    # success-count histogram in addition to marginal response accuracy.
    if success_counts:
        rollout_n = max(int((uids == uid).sum()) for uid in np.unique(uids))
        for successes in range(rollout_n + 1):
            metrics[f"batch/success_count_{successes}"] = float(
                success_counts.count(successes)
            )
        success_count_tensor = torch.tensor(success_counts, dtype=torch.float32)
        metrics["batch/success_count_mean"] = float(success_count_tensor.mean().item())
        metrics["batch/success_count_std"] = float(
            success_count_tensor.std(unbiased=False).item()
        )
    return metrics


def compute_on_off_metrics(
    batch: DataProto,
    reward_tensor: torch.Tensor,
    prefix_mask: torch.Tensor,
) -> dict[str, float]:
    """Compute response statistics split by on/off-policy trajectory."""
    off_policy_mask = prefix_mask.any(dim=-1)
    on_policy_mask = ~off_policy_mask
    response_length = batch.batch["response_mask"].sum(dim=-1)
    sequence_score = reward_tensor.sum(dim=-1)

    return {
        "on_off_metrics/on_response_length_mean": _masked_mean(
            response_length, on_policy_mask
        ),
        "on_off_metrics/off_response_length_mean": _masked_mean(
            response_length, off_policy_mask
        ),
        "on_off_metrics/on_score": _masked_mean(sequence_score, on_policy_mask),
        "on_off_metrics/off_score": _masked_mean(sequence_score, off_policy_mask),
        "on_off_metrics/off_example_ratio": off_policy_mask.float().mean().item(),
    }


def compute_rollout_diversity_metrics(
    batch: DataProto, uids: np.ndarray
) -> dict[str, float]:
    """Measure distinct generated responses within each GRPO prompt group."""
    responses = batch.batch["responses"]
    response_mask = batch.batch["response_mask"]
    unique_counts = []
    for uid in np.unique(uids):
        signatures = set()
        for row in np.flatnonzero(uids == uid).tolist():
            length = int(response_mask[row].sum().item())
            token_ids = responses[row, :length].detach().cpu().tolist()
            signatures.add(tuple(token_ids))
        unique_counts.append(len(signatures))

    rollout_n = max((int((uids == uid).sum()) for uid in np.unique(uids)), default=0)
    total_responses = len(uids)
    total_unique = sum(unique_counts)
    return {
        "batch/unique_responses_per_group": float(np.mean(unique_counts))
        if unique_counts
        else 0.0,
        "batch/unique_response_ratio": float(total_unique / total_responses)
        if total_responses
        else 0.0,
        "batch/duplicate_response_groups": float(
            sum(count < rollout_n for count in unique_counts)
        ),
    }


def compute_grpo_rollout_metrics(
    batch: DataProto,
    reward_tensor: torch.Tensor,
    config: DictConfig | dict | None = None,
) -> dict[str, float]:
    """Compute HPT-compatible metrics for an entirely on-policy GRPO batch."""
    config = config or {}
    uids = batch.non_tensor_batch["uid"]
    metrics = compute_grpo_batch_metrics(
        reward_tensor.sum(dim=-1),
        uids,
        success_reward=float(_get(config, "success_reward", 1.0)),
        failure_reward=float(_get(config, "failure_reward", 0.0)),
        format_reward=float(_get(config, "format_reward", -1.0)),
    )
    metrics.update(compute_rollout_diversity_metrics(batch, uids))
    prefix_mask = torch.zeros_like(batch.batch["response_mask"], dtype=torch.bool)
    metrics.update(compute_on_off_metrics(batch, reward_tensor, prefix_mask))
    return metrics
