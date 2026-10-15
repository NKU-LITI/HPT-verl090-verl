# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os

from transformers import PreTrainedTokenizerBase, ProcessorMixin

from .tokenizer import normalize_token_ids

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def initialize_system_prompt(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    """
    Initialize system prompt tokens for chat templates that support them.

    Args:
        tokenizer: The tokenizer with a chat template
        **apply_chat_template_kwargs: Additional arguments for apply_chat_template

    Returns:
        List of token IDs for the system prompt, or empty list if not supported
    """
    token1 = normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True, **apply_chat_template_kwargs
        )
    )
    token2 = normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}] * 2,
            add_generation_prompt=False,
            tokenize=True,
            **apply_chat_template_kwargs,
        )
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    return system_prompt


def initialize_turn_separator(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    """Tokens a chat template inserts after a message's closing token, before the next turn.

    Multi-turn agent rollouts build the token sequence incrementally. The model stops at the
    assistant close token (e.g. ``<|im_end|>``) and never emits the template's trailing
    turn-separator (e.g. ``"\\n"``, id 198 for Qwen). Rendering the following tool/user turn in
    isolation also omits that separator, so every turn boundary silently drops it and the rollout
    token sequence diverges from ``apply_chat_template`` of the equivalent full conversation.
    This returns the separator so callers can restore it at turn boundaries.

    Derivation: rendering the same (user) turn with empty vs non-empty content only differs in the
    content region, so the maximal common trailing run is exactly ``[close_token, *separator]``.
    A user turn is used deliberately -- probing with an assistant turn would inject reasoning
    scaffolding (e.g. Qwen3's ``<think></think>``) that is not part of the separator. The model
    emits the close token itself (it is the stop token), so the separator is everything after it.

    Returns an empty list when the template has no turn separator or an unexpected structure, so
    callers keep their previous behavior instead of crashing.
    """
    # Render two user turns that differ only in body text; the shared trailing run is the separator.
    # A bare string ``content`` is rejected by some multimodal processors (they iterate ``content``
    # expecting a list of typed parts), so fall back to the list-of-parts form, and return ``[]`` if
    # neither renders. Both probes must use the same form so only the body differs.
    empty = filled = None
    for as_parts in (False, True):
        if as_parts:
            body_empty, body_filled = [{"type": "text", "text": ""}], [{"type": "text", "text": "x"}]
        else:
            body_empty, body_filled = "", "x"
        try:
            empty = normalize_token_ids(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": body_empty}],
                    add_generation_prompt=False,
                    tokenize=True,
                    **apply_chat_template_kwargs,
                )
            )
            filled = normalize_token_ids(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": body_filled}],
                    add_generation_prompt=False,
                    tokenize=True,
                    **apply_chat_template_kwargs,
                )
            )
            break
        except Exception:
            empty = filled = None
    if empty is None or filled is None:
        return []
    # Maximal common trailing run == the message closing token(s) + inter-turn separator (identical
    # regardless of content).
    i = 0
    while i < len(empty) and i < len(filled) and empty[-1 - i] == filled[-1 - i]:
        i += 1
    suffix = empty[len(empty) - i :]
    if not suffix:
        return []
    # Split off the closing token the model already emits; the remainder is the dropped separator.
    # A processor (VLM path) exposes ``eos_token_id`` on its wrapped tokenizer rather than itself,
    # and some tokenizers (e.g. Llama 3) expose it as a list/tuple of ids rather than a single int.
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_id = getattr(getattr(tokenizer, "tokenizer", None), "eos_token_id", None)
    eos_ids = {eos_id} if isinstance(eos_id, int) else set(eos_id or [])
    last_close = max((i for i, tok_id in enumerate(suffix) if tok_id in eos_ids), default=None)
    if last_close is not None:
        return suffix[last_close + 1 :]
    return suffix[1:]


def extract_system_prompt_and_generation(tokenizer, **apply_chat_template_kwargs):
    token1 = normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True, **apply_chat_template_kwargs
        )
    )
    token2 = normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}] * 2,
            add_generation_prompt=False,
            tokenize=True,
            **apply_chat_template_kwargs,
        )
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    # get generate prompt tokens
    token3 = normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True, **apply_chat_template_kwargs
        )
    )
    generate_prompt = token3[len(token1) :]

    return system_prompt, generate_prompt


# def apply_chat_template(
#     processor: PreTrainedTokenizerBase | ProcessorMixin,
#     messages: list[dict],
#     *,
#     tokenize: bool = True,
#     add_generation_prompt: bool = True,
#     tools=None,
#     return_dict: bool = False,
#     **kwargs,
# ) -> list[int] | str:
#     """apply_chat_template to messages with special attention to template requiring
#     at least one user message, e.g. Qwen3.5.

#     Args:
#         processor: tokenizer or processor.
#         messages: list[dict], messages.
#         tokenize: bool, whether to tokenize the output.
#         add_generation_prompt: bool, whether to add generation prompt.
#         tools: list[dict], tools schema.
#         return_dict: bool, whether to return a dict.
#         **kwargs: additional arguments for apply_chat_template.

#     Returns:
#         list[int] | str: tokenized ids or text string.
#     """
#     try:
#         return processor.apply_chat_template(
#             messages,
#             tokenize=tokenize,
#             add_generation_prompt=add_generation_prompt,
#             tools=tools,
#             return_dict=return_dict,
#             **kwargs,
#         )
#     except Exception:
#         # Qwen3.5 apply_chat_template needs messages with at least one user message
#         dummy_user_message = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
#         dummy_user_prefix = processor.apply_chat_template(
#             dummy_user_message,
#             tokenize=tokenize,
#             add_generation_prompt=False,
#             tools=tools,
#             return_dict=return_dict,
#             **kwargs,
#         )
#         output = processor.apply_chat_template(
#             dummy_user_message + messages,
#             tokenize=tokenize,
#             add_generation_prompt=add_generation_prompt,
#             tools=tools,
#             return_dict=return_dict,
#             **kwargs,
#         )

#         if not tokenize:  # tokenize=False
#             return output[len(dummy_user_prefix) :]
#         elif not return_dict:  # tokenize=True and return_dict=False
#             if isinstance(output[0], list):  # transformers>=5
#                 assert len(output) == 1, "output must be a list[int] or list[list[int]]"
#                 dummy_user_prefix = dummy_user_prefix[0]
#                 output = output[0]
#             return output[len(dummy_user_prefix) :]
#         else:  # tokenize=True and return_dict=True and return_tensors="pt"
#             dummy_user_prefix = dict(dummy_user_prefix)
#             output = dict(output)
#             prefix_len = dummy_user_prefix["input_ids"].shape[1]
#             output["input_ids"] = output["input_ids"][:, prefix_len:]
#             output["attention_mask"] = output["attention_mask"][:, prefix_len:]
#             if "mm_token_type_ids" in output:
#                 output["mm_token_type_ids"] = output["mm_token_type_ids"][:, prefix_len:]
#             return output

def _rendered_len(output, *, tokenize: bool, return_dict: bool) -> int:
    """apply_chat_template 输出的长度（字符数或 token 数）"""
    if not tokenize:
        return len(output)
    if not return_dict:
        if isinstance(output[0], list):  # transformers>=5
            assert len(output) == 1, "output must be a list[int] or list[list[int]]"
            output = output[0]
        return len(output)
    # tokenize=True and return_dict=True and return_tensors="pt"
    return dict(output)["input_ids"].shape[1]


def _slice_rendered(output, start: int, end: int | None, *, tokenize: bool, return_dict: bool):
    """按位置切片 apply_chat_template 的输出，三种返回形式统一处理"""
    if not tokenize:
        return output[start:end]
    if not return_dict:
        if isinstance(output[0], list):  # transformers>=5
            assert len(output) == 1, "output must be a list[int] or list[list[int]]"
            output = output[0]
        return output[start:end]
    output = dict(output)
    for k in ("input_ids", "attention_mask", "mm_token_type_ids"):
        if k in output:
            output[k] = output[k][:, start:end]
    return output


def apply_chat_template(
    processor: PreTrainedTokenizerBase | ProcessorMixin,
    messages: list[dict],
    *,
    tokenize: bool = True,
    add_generation_prompt: bool = True,
    tools=None,
    return_dict: bool = False,
    **kwargs,
) -> list[int] | str:
    """apply_chat_template to messages with special attention to template requiring
    at least one user message and system-first ordering, e.g. Qwen3.5.
    """
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )
    except Exception:
        common = dict(tokenize=tokenize, tools=tools, return_dict=return_dict, **kwargs)
        fmt = dict(tokenize=tokenize, return_dict=return_dict)
        dummy_user_message = [{"role": "user", "content": [{"type": "text", "text": ""}]}]

        if messages and messages[0].get("role") == "system":
            # Qwen3.5: system 必须在第一位，且整段对话里至少要有一个 user。
            # 不能像下面那样在前面插 dummy user，改为在后面追加：
            #   one = [system, u]，two = [system, u, u]
            # 非首位的 user turn 渲染结果与位置无关，所以 len(two) - len(one) 就是一个 user turn 的长度，
            # 从 one 尾部截掉这段，剩下的就是 system 单独渲染的结果。
            # 用差分而不是单独渲染 [u] 求后缀，可以避免模板在"首条是 user"时插入默认 system 造成误差；
            # tools 会被渲染进开头的 system 块，因此带 tools 时同样成立。
            # 注意：这条路径忽略 add_generation_prompt（单独渲染 system 时它没有意义）。
            one = processor.apply_chat_template(messages + dummy_user_message, add_generation_prompt=False, **common)
            two = processor.apply_chat_template(messages + dummy_user_message * 2, add_generation_prompt=False, **common)
            len_one = _rendered_len(one, **fmt)
            turn_len = _rendered_len(two, **fmt) - len_one
            assert turn_len > 0, "failed to infer user-turn length for system-only rendering"
            return _slice_rendered(one, 0, len_one - turn_len, **fmt)

        # 原逻辑：Qwen3.5 需要至少一个 user，在前面插 dummy user 再截掉前缀（适用于 assistant/tool 消息）
        dummy_user_prefix = processor.apply_chat_template(dummy_user_message, add_generation_prompt=False, **common)
        output = processor.apply_chat_template(
            dummy_user_message + messages, add_generation_prompt=add_generation_prompt, **common
        )
        return _slice_rendered(output, _rendered_len(dummy_user_prefix, **fmt), None, **fmt)