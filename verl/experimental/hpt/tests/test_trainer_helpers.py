import pytest

from verl.trainer.ppo.ray_trainer import _hpt_legacy_gradient_accumulation, _mean_worker_metric


def test_mean_worker_metric_accepts_collected_worker_list():
    metrics = [{"mfu": 0.1}, {"mfu": [0.2, 0.3]}]
    assert _mean_worker_metric(metrics, "mfu") == pytest.approx(0.2)


def test_hpt_gradient_accumulation_accepts_per_gpu_micro_batch_size():
    actor_config = {
        "ppo_mini_batch_size": 6,
        "ppo_micro_batch_size": None,
        "ppo_micro_batch_size_per_gpu": 1,
    }
    assert _hpt_legacy_gradient_accumulation(actor_config, dp_size=48, rollout_n=8) == 1


def test_hpt_gradient_accumulation_preserves_legacy_global_config():
    actor_config = {
        "ppo_mini_batch_size": 6,
        "ppo_micro_batch_size": 2,
        "ppo_micro_batch_size_per_gpu": None,
    }
    assert _hpt_legacy_gradient_accumulation(actor_config, dp_size=48, rollout_n=8) == 3
