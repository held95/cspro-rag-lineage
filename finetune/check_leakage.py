"""Leakage and validity checks for the fine-tune dataset. Fails loudly.

    python -m finetune.check_leakage            # exit code 1 and a list of problems on failure

Between every pair of splits (train / val / heldout / heldout_ood) it checks: shared
scenario family, tenant id, doc id, entity string or fact value; duplicate user prompts
(the per-example nonce is masked first); and the mean 8-gram overlap of the later split's
prompts against the earlier split. Every completion must parse as ``GroundedAnswer`` and
every citation quote must be a verbatim substring (``find_quote``) of the cited source.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from pydantic import ValidationError

from app.modules.generation.normalize import find_quote
from app.schemas.ask import GroundedAnswer

DATA_DIR = Path(__file__).parent / "data"
NGRAM_N = 8
NGRAM_THRESHOLD = 0.30  # mean share of a held-out prompt's 8-grams also seen in train
META_KEYS = ("tenant_id", "doc_ids", "entities", "fact_values")


class LeakageError(RuntimeError):
    pass


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def user_text(example: dict) -> str:
    """The user message with the random nonce masked (so near-duplicates are detected)."""
    text = example["prompt"][-1]["content"]
    return text.replace(example["meta"]["nonce"], "<NONCE>")


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def ngram_overlap(later: list[dict], earlier: list[dict], n: int = NGRAM_N) -> float:
    """Mean fraction of each ``later`` prompt's n-grams that also occur in ``earlier``."""
    seen: set[tuple[str, ...]] = set()
    for ex in earlier:
        seen |= _ngrams(user_text(ex), n)
    ratios = []
    for ex in later:
        grams = _ngrams(user_text(ex), n)
        if grams:
            ratios.append(len(grams & seen) / len(grams))
    return sum(ratios) / len(ratios) if ratios else 0.0


def _values(examples: list[dict], key: str) -> set[str]:
    out: set[str] = set()
    for ex in examples:
        value = ex["scenario_family"] if key == "scenario_family" else ex["meta"][key]
        out |= set(value) if isinstance(value, list) else {value}
    return out


def check_pair(name_a: str, a: list[dict], name_b: str, b: list[dict],
               threshold: float = NGRAM_THRESHOLD) -> list[str]:
    """Problems found between two splits (empty list = clean)."""
    if not a or not b:
        return []
    problems = []
    for key in ("scenario_family",) + META_KEYS:
        shared = _values(a, key) & _values(b, key)
        if shared:
            problems.append(f"{name_a}/{name_b} share {key}: {sorted(shared)[:5]}")
    dupes = {user_text(x) for x in a} & {user_text(x) for x in b}
    if dupes:
        problems.append(f"{name_a}/{name_b} duplicate prompt(s): {len(dupes)}")
    overlap = ngram_overlap(b, a)
    if overlap > threshold:
        problems.append(f"{name_a}/{name_b} mean {NGRAM_N}-gram overlap {overlap:.2f} > {threshold}")
    return problems


def validate_examples(examples: list[dict]) -> list[str]:
    """Every completion parses as GroundedAnswer; every quote is verbatim in its source."""
    problems = []
    for ex in examples:
        try:
            answer = GroundedAnswer.model_validate_json(ex["completion"])
        except ValidationError as err:
            problems.append(f"{ex['id']}: completion is not a GroundedAnswer ({err.error_count()} errors)")
            continue
        sources = ex["meta"]["sources"]
        for cit in answer.citations:
            if cit.source_id not in sources:
                problems.append(f"{ex['id']}: cites unknown source {cit.source_id}")
            elif find_quote(sources[cit.source_id], cit.quote) is None:
                problems.append(f"{ex['id']}: quote not verbatim in {cit.source_id}")
        if ex["meta"]["canary"] and ex["meta"]["canary"] in ex["completion"]:
            problems.append(f"{ex['id']}: completion leaks the canary")
    return problems


def find_problems(train: list[dict], val: list[dict], heldout: list[dict],
                  ood: list[dict]) -> list[str]:
    splits = {"train": train, "val": val, "heldout": heldout, "heldout_ood": ood}
    names = list(splits)
    problems = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            problems += check_pair(a, splits[a], b, splits[b])
    for name, rows in splits.items():
        dup_ids = len(rows) - len({r["id"] for r in rows})
        if dup_ids:
            problems.append(f"{name}: {dup_ids} duplicate id(s)")
        problems += validate_examples(rows)
    return problems


def assert_no_leakage(train, val, heldout, ood) -> None:
    problems = find_problems(train, val, heldout, ood)
    if problems:
        raise LeakageError("\n".join(problems))


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail if fine-tune splits leak into each other.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    d = args.data_dir
    splits = [load_jsonl(d / f) for f in
              ("train.jsonl", "val.jsonl", "heldout.jsonl", "heldout_ood_handwritten.jsonl")]
    problems = find_problems(*splits)
    for p in problems:
        print("LEAK/INVALID:", p)
    if problems:
        return 1
    print(f"OK: no leakage; train/heldout mean {NGRAM_N}-gram overlap "
          f"{ngram_overlap(splits[2], splits[0]):.3f} (threshold {NGRAM_THRESHOLD}); "
          f"ood {ngram_overlap(splits[3], splits[0]):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
