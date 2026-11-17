"""DATA-02: the corpus and the fine-tune data are synthetic only (no real e-mails, phones, orgs or key-like strings)."""
import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
FILES = [ROOT / "data" / "synthetic" / "records.json", *sorted((ROOT / "finetune" / "data").glob("*.jsonl"))]

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[ -])?\(?\d{2,4}\)?[ -]\d{3,4}[ -]\d{3,4}(?!\d)")
KEYLIKE = re.compile(r"sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{20,}|"
                     r"BEGIN (?:RSA |OPENSSH )?PRIVATE KEY|eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}")
URL = re.compile(r"https?://([^/\s\"')]+)")
ALLOWED_HOSTS = ("example.com", "example.org", "example.net", ".example", "localhost")


def test_data_files_exist():
    assert FILES[0].exists() and len(FILES) >= 5


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_data02_no_emails_phones_keys_or_real_hosts(path):
    text = path.read_text(encoding="utf-8")
    assert not EMAIL.findall(text), f"e-mail address in {path.name}"
    assert not PHONE.findall(text), f"phone-like number in {path.name}"
    assert not KEYLIKE.findall(text), f"key-like string in {path.name}"
    hosts = {h.lower() for h in URL.findall(text)}
    bad = {h for h in hosts if not h.endswith(ALLOWED_HOSTS)}
    assert not bad, f"non-reserved hosts in {path.name}: {sorted(bad)}"


def test_records_json_is_well_formed_and_has_the_expected_roles():
    records = json.loads(FILES[0].read_text(encoding="utf-8"))
    assert {r["tenant"] for r in records} == {"alpha", "beta"}
    assert len({(r["tenant"], r["doc_id"]) for r in records}) == 14  # 14 documents (15 versions: KB-001 has two)
    assert any("SYSTEM:" in r["content"] for r in records)        # forged-instruction record
    assert any(r["acl_groups"] for r in records)                   # restricted record
