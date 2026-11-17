"""QLoRA supervised fine-tune with TRL's SFTTrainer.

    python -m finetune.train --config finetune/configs/qlora_qwen3_1p7b.yaml [--max-steps 20]

Data are prompt-completion pairs: the prompt is rendered with ``render.render_prompt`` (the
same function generation uses) and the loss is computed on completion tokens only
(``completion_only_loss=True``). Heavy libraries are imported inside ``run_training`` so the
helpers below can be unit-tested without torch.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from finetune.render import render_prompt

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "model": {"name": str, "revision": str},
    "data": {"train_path": str, "val_path": str},
    "lora": {"r": int, "alpha": int, "dropout": float, "target_modules": list},
    "train": {"max_length": int, "per_device_train_batch_size": int,
              "gradient_accumulation_steps": int, "num_train_epochs": (int, float),
              "learning_rate": float, "lr_scheduler_type": str, "warmup_ratio": float,
              "optim": str, "seed": int, "logging_steps": int, "eval_steps": int,
              "save_steps": int, "gradient_checkpointing": bool},
    "output_dir": str,
}
LIBS = ("torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate", "datasets")


class ConfigError(ValueError):
    pass


# ----------------------------------------------------------------- config + data helpers

def validate_config(cfg: dict) -> dict:
    """Check every required key exists with the right type; the revision must be a commit hash."""
    for section, spec in REQUIRED.items():
        if section not in cfg:
            raise ConfigError(f"missing section: {section}")
        if not isinstance(spec, dict):
            if not isinstance(cfg[section], spec):
                raise ConfigError(f"{section} must be {spec}")
            continue
        for key, typ in spec.items():
            if key not in cfg[section]:
                raise ConfigError(f"missing key: {section}.{key}")
            if not isinstance(cfg[section][key], typ):
                raise ConfigError(f"{section}.{key} must be {typ}")
    rev = cfg["model"]["revision"]
    if len(rev) != 40 or any(c not in "0123456789abcdef" for c in rev):
        raise ConfigError(f"model.revision must be a 40-char commit hash, got {rev!r}")
    return cfg


def load_config(path: str | Path) -> dict:
    import yaml  # imported here so the rest of the module works without pyyaml

    with open(path, encoding="utf-8") as fh:
        return validate_config(yaml.safe_load(fh))


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_sft_rows(examples: list[dict], tokenizer) -> list[dict]:
    """Training-path rendering: prompt text (shared renderer) + gold JSON + end-of-turn token."""
    return [{"prompt": render_prompt(tokenizer, ex["prompt"]),
             "completion": ex["completion"] + tokenizer.eos_token} for ex in examples]


def count_trainable_tokens(tokenizer, row: dict, max_length: int) -> int:
    """Completion tokens that survive truncation to ``max_length`` (= tokens with loss)."""
    n_prompt = len(tokenizer(row["prompt"], add_special_tokens=False)["input_ids"])
    n_full = len(tokenizer(row["prompt"] + row["completion"], add_special_tokens=False)["input_ids"])
    return max(0, min(n_full, max_length) - n_prompt)


def _git_sha() -> str | None:
    """Read the current commit from .git without running git."""
    head = ROOT / ".git" / "HEAD"
    if not head.exists():
        return None
    ref = head.read_text().strip()
    if not ref.startswith("ref: "):
        return ref
    ref_file = ROOT / ".git" / ref[5:]
    if ref_file.exists():
        return ref_file.read_text().strip()
    packed = ROOT / ".git" / "packed-refs"
    for line in packed.read_text().splitlines() if packed.exists() else []:
        if line.endswith(ref[5:]):
            return line.split()[0]
    return None


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for lib in LIBS:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            out[lib] = None
    return out


# ----------------------------------------------------------------- training (GPU)

def run_training(cfg: dict, output_dir: Path, max_steps: int | None) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    t = cfg["train"]
    name, revision = cfg["model"]["name"], cfg["model"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    train_rows = build_sft_rows(load_jsonl(ROOT / cfg["data"]["train_path"]), tokenizer)
    val_rows = build_sft_rows(load_jsonl(ROOT / cfg["data"]["val_path"]), tokenizer)
    trainable = [count_trainable_tokens(tokenizer, r, t["max_length"]) for r in train_rows]
    if min(trainable) == 0:
        raise RuntimeError("some training rows have no completion tokens after truncation")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, quantization_config=bnb, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": 0})
    lora = LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
                      lora_dropout=cfg["lora"]["dropout"],
                      target_modules=cfg["lora"]["target_modules"], task_type="CAUSAL_LM")
    args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        max_length=t["max_length"],
        completion_only_loss=True,
        packing=False,
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        num_train_epochs=t["num_train_epochs"],
        max_steps=max_steps if max_steps else -1,
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_steps=t["warmup_ratio"],  # transformers>=5: a float in [0,1) means "fraction of total steps"
        optim=t["optim"],
        seed=t["seed"],
        bf16=True,
        gradient_checkpointing=t["gradient_checkpointing"],
        logging_steps=t["logging_steps"],
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        save_strategy="steps",
        save_steps=t["save_steps"],
        save_total_limit=2,
        dataloader_num_workers=0,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(train_rows),
                         eval_dataset=Dataset.from_list(val_rows), processing_class=tokenizer,
                         peft_config=lora)
    # TRL >= 1.x folds completion_mask into `labels` (-100 = no loss); count the tokens that DO train
    first_labels = trainer.train_dataset[0]["labels"]
    first_mask = [1 if tok != -100 else 0 for tok in first_labels]
    if sum(first_mask) == 0:
        raise RuntimeError("completion mask is all zeros: nothing would be trained")
    if sum(first_mask) == len(first_mask):
        raise RuntimeError("no prompt tokens are masked: loss would include the prompt (completion_only_loss off?)")

    torch.cuda.reset_peak_memory_stats()
    started = datetime.now(timezone.utc)
    t0 = time.time()
    result = trainer.train()
    wall = time.time() - t0
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))

    steps = result.global_step
    return {
        "model": name, "revision": revision, "git_sha": _git_sha(), "versions": _versions(),
        "seed": t["seed"], "gpu": torch.cuda.get_device_name(0),
        "started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_time_s": round(wall, 1), "global_steps": steps,
        "seconds_per_step": round(wall / steps, 2) if steps else None,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
        "final_train_loss": result.training_loss,
        "trainable_tokens_min_max": [min(trainable), max(trainable)],
        "first_row_completion_tokens": int(sum(first_mask)),
        "max_steps_override": max_steps, "config": cfg,
        "log_history": trainer.state.log_history, "adapter_dir": str(adapter_dir),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QLoRA SFT (prompt-completion, completion-only loss).")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=None, help="e.g. 20 for the smoke run")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    output_dir = args.output_dir or ROOT / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = run_training(cfg, output_dir, args.max_steps)
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n",
                                              encoding="utf-8")
    if not math.isfinite(meta["final_train_loss"]):
        raise RuntimeError(f"non-finite training loss: {meta['final_train_loss']}")
    print(f"done: {meta['global_steps']} steps, {meta['wall_time_s']}s, "
          f"peak VRAM {meta['peak_vram_gb']} GB, loss {meta['final_train_loss']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
