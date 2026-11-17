"""FT-10 train prompt == eval prompt, FT-08 config validates, FT-04 loss mask non-zero,
FT-09 20-step GPU smoke run (``-m gpu``, not run by default)."""
import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from finetune import gen_dataset, generate, render, train

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = sorted((ROOT / "finetune" / "configs").glob("*.yaml"))


class FakeQwen3Tokenizer:
    """Mimics the Qwen3 chat template, including the empty think block when thinking is off."""
    eos_token = "<|im_end|>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            enable_thinking=True):
        assert tokenize is False
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
            if not enable_thinking:
                text += "<think>\n\n</think>\n\n"
        return text


@pytest.fixture(scope="module")
def example():
    return gen_dataset.generate_split("train", 5, seed=42)[3]


@pytest.mark.unit
def test_ft10_train_and_eval_render_identical_prompts(example):
    tok = FakeQwen3Tokenizer()
    train_prompt = train.build_sft_rows([example], tok)[0]["prompt"]
    eval_prompt = generate.prompt_text(example, tok)
    assert train_prompt.encode("utf-8") == eval_prompt.encode("utf-8")
    assert eval_prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert example["prompt"][1]["content"] in eval_prompt


@pytest.mark.unit
def test_sft_completion_is_gold_json_plus_eos(example):
    row = train.build_sft_rows([example], FakeQwen3Tokenizer())[0]
    assert row["completion"] == example["completion"] + "<|im_end|>"


def _cached_tokenizer():
    transformers = pytest.importorskip("transformers", reason="transformers not installed (train venv only)")
    cfg = train.load_config(ROOT / "finetune" / "configs" / "qlora_qwen3_1p7b.yaml")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            cfg["model"]["name"], revision=cfg["model"]["revision"], local_files_only=True), cfg
    except OSError as err:
        pytest.skip(f"tokenizer for {cfg['model']['name']} not in the local HF cache: {err}")


@pytest.mark.unit
def test_ft10_real_qwen3_template_identical_and_has_empty_think_block(example):
    tok, _ = _cached_tokenizer()
    a = train.build_sft_rows([example], tok)[0]["prompt"]
    b = generate.prompt_text(example, tok)
    assert a == b
    assert a.endswith("<think>\n\n</think>\n\n")


@pytest.mark.unit
def test_ft04_loss_mask_has_trainable_tokens():
    tok, cfg = _cached_tokenizer()
    rows = train.build_sft_rows(gen_dataset.generate_split("train", 200, seed=42), tok)
    counts = [train.count_trainable_tokens(tok, r, cfg["train"]["max_length"]) for r in rows]
    assert min(counts) > 0


@pytest.mark.unit
@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_ft08_configs_load_and_validate(path):
    pytest.importorskip("yaml", reason="pyyaml not installed in this venv")
    cfg = train.load_config(path)
    assert cfg["lora"]["r"] == 16 and cfg["lora"]["alpha"] == 32
    assert len(cfg["model"]["revision"]) == 40  # pinned commit hash
    assert cfg["train"]["max_length"] == 1536


@pytest.mark.unit
def test_ft08_invalid_config_is_rejected():
    pytest.importorskip("yaml", reason="pyyaml not installed in this venv")
    cfg = train.load_config(CONFIGS[0])
    broken = copy.deepcopy(cfg)
    del broken["lora"]["r"]
    with pytest.raises(train.ConfigError):
        train.validate_config(broken)
    broken = copy.deepcopy(cfg)
    broken["model"]["revision"] = "main"
    with pytest.raises(train.ConfigError):
        train.validate_config(broken)


@pytest.mark.gpu
@pytest.mark.slow
def test_ft09_twenty_step_smoke_train(tmp_path, request):
    if "gpu" not in request.config.getoption("markexpr"):
        pytest.skip("GPU smoke run only with `-m gpu` (needs the train venv and a free GPU)")
    torch = pytest.importorskip("torch", reason="torch not installed (train venv only)")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    out = tmp_path / "smoke"
    cmd = [sys.executable, "-m", "finetune.train", "--config",
           str(ROOT / "finetune" / "configs" / "qlora_qwen3_1p7b.yaml"),
           "--max-steps", "20", "--output-dir", str(out)]
    subprocess.run(cmd, cwd=ROOT, check=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert (out / "adapter" / "adapter_model.safetensors").exists()
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert math.isfinite(meta["final_train_loss"])
