"""The ONE function that turns an example's chat messages into model input text.

Both training (``train.build_sft_rows``) and generation (``generate.prompt_text``) call
``render_prompt``, so the two prompts are byte-identical (test FT-10). Qwen3 thinking is
disabled: with ``enable_thinking=False`` the template appends an empty
``<think>\\n\\n</think>\\n\\n`` block after the assistant header, and that block is part of
the prompt in BOTH paths.
"""
from __future__ import annotations


def render_prompt(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
