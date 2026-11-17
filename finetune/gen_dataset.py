"""Seeded, fully synthetic fine-tune dataset.

    python -m finetune.gen_dataset --seed 42

Each example is the SAME prompt the API sends (``build_messages``) plus a completion
that is the compact JSON of ``GroundedAnswer``. Four types, cycled 2:1:1:1:
answerable (40%), unanswerable (20%), conflict (20%), injection (20%).
Splits are assigned by scenario family (see ``scenarios.SPLIT_FAMILIES``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from app.modules.generation.prompt import PromptSource, build_messages
from app.schemas.ask import GroundedAnswer, LLMCitation
from finetune import scenarios
from finetune.ood_items import OOD_ITEMS
from finetune.scenarios import Family

SPLIT_SIZES = {"train": 200, "val": 40, "heldout": 60}
TYPE_CYCLE = ("answerable", "answerable", "unanswerable", "conflict", "injection")
DATA_DIR = Path(__file__).parent / "data"


class _Ctx:
    """Per-split random state and entity pools (keeps splits disjoint by construction)."""

    def __init__(self, split: str, seed: int):
        self.split = split
        self.rng = random.Random(f"{seed}-{split}")
        self.doc_counter = 0

    def doc_id(self) -> str:
        self.doc_counter += 1
        return f"{scenarios.DOC_ID_PREFIX[self.split]}-KB-{self.doc_counter:05d}"

    def org(self) -> str:
        return f"{self.rng.choice(scenarios.ORG_PREFIXES[self.split])} {self.rng.choice(scenarios.ORG_SUFFIXES)}"

    def value(self, fam: Family, exclude: int | None = None) -> int:
        pool = [v for v in scenarios.NUMBER_POOLS[self.split] if v != exclude]
        return self.rng.choice(pool)


def _doc(ctx: _Ctx, fam: Family, org: str, value: int | None, title_suffix: str = "") -> dict:
    """A policy-like document; ``value=None`` leaves out the answering sentence."""
    lines = list(fam.filler)
    if value is not None:
        lines.insert(ctx.rng.randrange(len(lines) + 1), fam.fact.format(org=org, v=value))
    return {"doc_id": ctx.doc_id(), "kind": fam.kind,
            "title": fam.title.format(org=org) + title_suffix, "text": " ".join(lines)}


def _distractors(ctx: _Ctx, fam: Family, org: str, k: int) -> list[dict]:
    """Documents from OTHER families of the same split (never leak across splits)."""
    others = [f for f in scenarios.FAMILIES
              if f.name in scenarios.SPLIT_FAMILIES[ctx.split] and f.name != fam.name]
    return [_doc(ctx, f, org, ctx.value(f)) for f in ctx.rng.sample(others, min(k, len(others)))]


def _ticket(ctx: _Ctx, canary: str) -> tuple[dict, str]:
    person = ctx.rng.choice(scenarios.PEOPLE[ctx.split])
    injection = ctx.rng.choice(scenarios.INJECTIONS[ctx.split]).format(canary=canary)
    line = ctx.rng.choice(scenarios.TICKET_LINES[ctx.split])
    text = f"Reported by {person}. {line} {injection}"
    return {"doc_id": ctx.doc_id(), "kind": "ticket",
            "title": f"Ticket {ctx.rng.randint(1000, 9999)}", "text": text}, person


def _cite(doc_alias: dict[str, str], doc: dict, quote: str) -> LLMCitation:
    return LLMCitation(source_id=doc_alias[doc["doc_id"]], quote=quote)


def make_example(ctx: _Ctx, fam: Family, ex_type: str, index: int) -> dict:
    """Build one example of ``ex_type`` for family ``fam``."""
    rng = ctx.rng
    org = ctx.org()
    question = fam.question.format(org=org)
    canary, people, citing = "", [], []  # citing: (doc, quote) pairs for the gold answer
    values: list[int] = []
    docs: list[dict] = []
    answer_text = ""
    conflict = False

    # injection items: half still contain the real answer, half must be abstained on
    has_answer = ex_type == "answerable" or (ex_type == "injection" and rng.random() < 0.5)
    if has_answer:
        v = ctx.value(fam)
        values.append(v)
        main = _doc(ctx, fam, org, v)
        docs.append(main)
        citing.append((main, fam.fact.format(org=org, v=v)))
        answer_text = fam.claim.format(v=v).capitalize() + "."
        if ex_type == "answerable" and rng.random() < 0.3:  # second, restating source
            faq = {"doc_id": ctx.doc_id(), "kind": "faq", "title": f"{org} Help Centre",
                   "text": fam.faq.format(org=org, v=v)}
            docs.append(faq)
            citing.append((faq, faq["text"]))
    elif ex_type == "unanswerable":
        docs.append(_doc(ctx, fam, org, None))  # on-topic but without the fact
    elif ex_type == "conflict":
        v1 = ctx.value(fam)
        v2 = ctx.value(fam, exclude=v1)
        values += [v1, v2]
        a, b = _doc(ctx, fam, org, v1), _doc(ctx, fam, org, v2, " (revised draft)")
        docs += [a, b]
        citing += [(a, fam.fact.format(org=org, v=v1)), (b, fam.fact.format(org=org, v=v2))]
        conflict = True

    if ex_type == "injection":
        canary = f"CANARY-{rng.getrandbits(32):08x}"
        ticket, person = _ticket(ctx, canary)
        docs.append(ticket)
        people.append(person)

    docs += _distractors(ctx, fam, org, rng.randint(0 if docs else 1, 2))
    rng.shuffle(docs)
    alias = {d["doc_id"]: f"S{i + 1}" for i, d in enumerate(docs)}

    citations = [_cite(alias, d, q) for d, q in citing]
    if conflict:  # list the lower alias first so the answer reads S1 ... S2 ...
        pairs = sorted(zip(citations, values), key=lambda cv: cv[0].source_id)
        citations = [c for c, _ in pairs]
        (c1, v1), (c2, v2) = pairs
        answer_text = (f"The sources conflict: {c1.source_id} says {fam.claim.format(v=v1)}, "
                       f"while {c2.source_id} says {fam.claim.format(v=v2)}.")
    gold = GroundedAnswer(answer=answer_text, abstain=not citations, conflict=conflict,
                          citations=citations)

    nonce = f"{rng.getrandbits(64):016x}"
    sources = [PromptSource(alias[d["doc_id"]], d["kind"], d["title"], d["text"]) for d in docs]
    return {
        "id": f"{ctx.split}-{index:04d}",
        "scenario_family": fam.name,
        "split": ctx.split,
        "prompt": build_messages(question, sources, nonce),
        "completion": gold.model_dump_json(),
        "meta": {
            "type": ex_type,
            "gold_answer": gold.answer,
            "gold_source_ids": [c.source_id for c in gold.citations],
            "canary": canary,
            "question": question,
            "nonce": nonce,
            "sources": {s.alias: s.text for s in sources},
            "tenant_id": "tenant-" + org.lower().replace(" ", "-"),
            "doc_ids": [d["doc_id"] for d in docs],
            "entities": [org] + people,
            "fact_values": [str(v) for v in values],
        },
    }


def generate_split(split: str, n: int, seed: int) -> list[dict]:
    ctx = _Ctx(split, seed)
    families = [scenarios.family_by_name(f) for f in scenarios.SPLIT_FAMILIES[split]]
    return [make_example(ctx, families[i % len(families)], TYPE_CYCLE[i % len(TYPE_CYCLE)], i)
            for i in range(n)]


def generate_all(seed: int) -> dict[str, list[dict]]:
    return {split: generate_split(split, n, seed) for split, n in SPLIT_SIZES.items()}


def build_ood_items() -> list[dict]:
    """Render the hand-written OOD items (``ood_items.py``) through the same prompt builder."""
    out = []
    for i, item in enumerate(OOD_ITEMS):
        sources = [PromptSource(f"S{j + 1}", s["kind"], s["title"], s["text"])
                   for j, s in enumerate(item["sources"])]
        gold = GroundedAnswer.model_validate(item["gold"])
        nonce = hashlib.sha256(f"ood-{i}".encode()).hexdigest()[:16]
        out.append({
            "id": f"ood-{i:04d}",
            "scenario_family": "ood_handwritten",
            "split": "heldout_ood",
            "prompt": build_messages(item["question"], sources, nonce),
            "completion": gold.model_dump_json(),
            "meta": {
                "type": item["type"],
                "gold_answer": gold.answer,
                "gold_source_ids": [c.source_id for c in gold.citations],
                "canary": item.get("canary", ""),
                "question": item["question"],
                "nonce": nonce,
                "sources": {s.alias: s.text for s in sources},
                "tenant_id": item["tenant_id"],
                "doc_ids": [f"OOD-{i:02d}-{j}" for j in range(len(sources))],
                "entities": item["entities"],
                "fact_values": [],
            },
        })
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> dict:
    data = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    path.write_bytes(data.encode("utf-8"))
    return {"count": len(rows), "sha256": hashlib.sha256(data.encode("utf-8")).hexdigest()}


def write_dataset(out_dir: Path, seed: int) -> dict:
    """Write all splits + MANIFEST.json into ``out_dir``; returns the manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {f"{split}.jsonl": _write_jsonl(out_dir / f"{split}.jsonl", rows)
             for split, rows in generate_all(seed).items()}
    files["heldout_ood_handwritten.jsonl"] = _write_jsonl(
        out_dir / "heldout_ood_handwritten.jsonl", build_ood_items())
    manifest = {"seed": seed, "split_families": scenarios.SPLIT_FAMILIES, "files": files}
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    manifest = write_dataset(args.out_dir, args.seed)
    for name, info in manifest["files"].items():
        print(f"{name}: {info['count']} rows sha256={info['sha256'][:12]}")


if __name__ == "__main__":
    main()
