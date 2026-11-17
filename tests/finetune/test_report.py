"""report.py: builds the markdown only from result files; missing arms say NOT EXECUTED."""
import json
import re

import pytest

from finetune import report

pytestmark = [pytest.mark.unit, pytest.mark.eval]

METRICS = {"schema_valid_rate": 0.9, "citation_quote_valid_rate": 0.8, "abstain_f1": 0.75,
           "answer_token_f1": None}


def _write(path, status="EXECUTED"):
    data = {"status": status, "n": 60, "metrics": METRICS if status == "EXECUTED" else None,
            "ci95": {k: [0.5, 0.95] if v is not None else None for k, v in METRICS.items()}}
    path.write_text(json.dumps(data), encoding="utf-8")


def test_missing_everything_reports_not_executed_without_numbers(tmp_path):
    md = report.build_report(tmp_path / "results", tmp_path / "run_meta.json")
    assert md.count("NOT EXECUTED") >= 4  # 2 arms x 2 evaluation sets
    assert not re.search(r"\d\.\d", md), "no metric values may appear"
    assert "run metadata: not available" in md


def test_executed_arm_shows_numbers_and_missing_arm_does_not(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    _write(results / "base__heldout.json")
    meta = tmp_path / "run_meta.json"
    meta.write_text(json.dumps({"model": "Qwen/Qwen3-1.7B", "revision": "abc", "wall_time_s": 12.5,
                                "global_steps": 75, "final_train_loss": 0.3}), encoding="utf-8")
    md = report.build_report(results, meta)
    heldout, ood = md.split("## Hand-written OOD")
    assert "0.900 [0.500, 0.950]" in heldout
    assert "n/a" in heldout  # None metric
    assert "NOT EXECUTED" in heldout  # adapted arm missing
    assert not re.search(r"\d\.\d{3}", ood)
    assert "Qwen/Qwen3-1.7B" in md and "n = 60" in md
