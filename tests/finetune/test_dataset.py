"""FT-01 (determinism), FT-02 (leakage checker catches a seeded leak), FT-03 (split disjointness)."""
import copy
import json

import pytest

from app.schemas.ask import GroundedAnswer
from finetune import check_leakage, gen_dataset

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def splits():
    return gen_dataset.generate_all(seed=42)


def test_ft01_generator_is_deterministic(tmp_path):
    m1 = gen_dataset.write_dataset(tmp_path / "a", seed=42)
    m2 = gen_dataset.write_dataset(tmp_path / "b", seed=42)
    assert m1["files"] == m2["files"]
    m3 = gen_dataset.write_dataset(tmp_path / "c", seed=7)
    assert m3["files"]["train.jsonl"]["sha256"] != m1["files"]["train.jsonl"]["sha256"]


def test_split_sizes_and_type_mix(splits):
    assert {k: len(v) for k, v in splits.items()} == {"train": 200, "val": 40, "heldout": 60}
    types = [ex["meta"]["type"] for ex in splits["train"]]
    assert types.count("answerable") == 80
    for t in ("unanswerable", "conflict", "injection"):
        assert types.count(t) == 40


def test_examples_have_required_fields_and_valid_completions(splits):
    for ex in splits["train"] + splits["heldout"]:
        assert set(ex) == {"id", "scenario_family", "split", "prompt", "completion", "meta"}
        assert [m["role"] for m in ex["prompt"]] == ["system", "user"]
        answer = GroundedAnswer.model_validate_json(ex["completion"])
        assert answer.abstain == (ex["meta"]["gold_answer"] == "")
        assert 1 <= len(ex["meta"]["sources"]) <= 5
        if ex["meta"]["canary"]:
            assert ex["meta"]["canary"] not in ex["completion"]
            assert ex["meta"]["canary"] in ex["prompt"][1]["content"]


def test_ft03_splits_are_disjoint(splits):
    def keys(split, field):
        out = set()
        for ex in splits[split]:
            value = ex["scenario_family"] if field == "family" else ex["meta"][field]
            out |= set(value) if isinstance(value, list) else {value}
        return out

    for field in ("family", "tenant_id", "doc_ids", "entities", "fact_values"):
        assert not keys("train", field) & keys("heldout", field), field
        assert not keys("train", field) & keys("val", field), field
        assert not keys("val", field) & keys("heldout", field), field


def test_real_dataset_passes_leakage_checks(splits):
    ood = gen_dataset.build_ood_items()
    assert len(ood) == 15
    assert check_leakage.find_problems(splits["train"], splits["val"], splits["heldout"], ood) == []


def test_ft02_leakage_checker_catches_seeded_leak(splits):
    heldout = copy.deepcopy(splits["heldout"])
    heldout.append(copy.deepcopy(splits["train"][0]))  # seeded leak: a train item in held-out
    problems = check_leakage.find_problems(splits["train"], splits["val"], heldout, [])
    text = "\n".join(problems)
    assert "scenario_family" in text
    assert "duplicate prompt" in text
    assert "doc_ids" in text
    with pytest.raises(check_leakage.LeakageError):
        check_leakage.assert_no_leakage(splits["train"], splits["val"], heldout, [])


def test_ft02_checker_catches_shared_entity_and_ngram_overlap(splits):
    heldout = copy.deepcopy(splits["heldout"])
    victim = heldout[0]
    victim["meta"]["entities"] = victim["meta"]["entities"] + [splits["train"][0]["meta"]["entities"][0]]
    problems = check_leakage.find_problems(splits["train"], [], heldout, [])
    assert any("entities" in p for p in problems)
    # near-duplicate text (different nonce) is caught by the n-gram check
    overlap = check_leakage.ngram_overlap(
        [splits["train"][1]], splits["train"][1:2], n=8)
    assert overlap == pytest.approx(1.0)


def test_checker_rejects_non_verbatim_quote(splits):
    bad = copy.deepcopy(splits["train"][0])
    answer = json.loads(bad["completion"])
    answer["citations"] = [{"source_id": "S1", "quote": "this sentence is not in any source"}]
    answer["abstain"] = False
    bad["completion"] = json.dumps(answer)
    problems = check_leakage.validate_examples([bad])
    assert any("quote" in p for p in problems)
