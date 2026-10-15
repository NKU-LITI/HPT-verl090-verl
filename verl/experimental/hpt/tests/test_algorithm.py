from types import SimpleNamespace

import torch

from verl.experimental.hpt.algorithm import _replace_with_target


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
