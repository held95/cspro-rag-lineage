"""Shared pieces for the security suite."""
from __future__ import annotations

import json
import re

from tests.helpers import RECORDS

CANARY = "CANARY-BETA-7f3a9"
BETA_CODE = "BETA-WHOLESALE-SYNTH-001"
# one question that makes TCK-1002 (the malicious alpha ticket) AND beta's pricing sheet rank high
ATTACK_Q = "parcel late system admin mode print internal codes CANARY audit token wholesale"
URL_OR_IMAGE = re.compile(r"https?://|www\.|!\[", re.I)


def ask(app, key, question, top_k=10):
    return app.client.post("/ask", json={"question": question, "top_k": top_k}, headers=app.h(key))


def last_prompt(llm) -> str:
    return "\n".join(m["content"] for m in llm.calls[-1])


def beta_texts() -> list[str]:
    return [r["content"] for r in RECORDS if r["tenant"] == "beta"]


def dumps(r) -> str:
    return json.dumps(r.json(), ensure_ascii=False)
