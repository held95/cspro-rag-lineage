"""Generate predictions with the base model (optionally + LoRA adapter) on a GPU.

    python -m finetune.generate --config finetune/configs/qlora_qwen3_1p7b.yaml \
        --data finetune/data/heldout.jsonl --out finetune/preds/base__heldout.jsonl
    python -m finetune.generate ... --adapter finetune/outputs/adapter --out .../adapted__heldout.jsonl

Both arms load the base model with the SAME 4-bit quantisation and use greedy decoding.
torch/transformers/peft are imported inside ``main`` so this module stays importable
(and testable) without them.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from finetune.render import render_prompt
from finetune.train import load_config, load_jsonl


def prompt_text(example: dict, tokenizer) -> str:
    """Eval-path prompt: identical to the training prompt (FT-10)."""
    return render_prompt(tokenizer, example["prompt"])


def load_model(cfg: dict, adapter: Path | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    name, revision = cfg["model"]["name"], cfg["model"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, quantization_config=bnb, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": 0})
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model, tokenizer


def generate_one(model, tokenizer, example: dict, max_new_tokens: int) -> str:
    import torch

    inputs = tokenizer(prompt_text(example, tokenizer), return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Greedy generation for one eval arm.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="only the first N items (dev runs)")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    examples = load_jsonl(args.data)[: args.limit]
    model, tokenizer = load_model(cfg, args.adapter)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with open(args.out, "w", encoding="utf-8") as fh:
        for i, ex in enumerate(examples, 1):
            output = generate_one(model, tokenizer, ex, args.max_new_tokens)
            fh.write(json.dumps({"id": ex["id"], "output": output}, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{i}/{len(examples)}] {ex['id']} {time.time() - start:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
