"""The JSON Schema handed to Ollama must force every key; otherwise a constrained model may omit
`citations` and the verifier abstains on every question (found in the live run, see docs/LIVE_RUN.md)."""
import pytest

from app.schemas.ask import answer_json_schema

pytestmark = pytest.mark.unit


def test_schema_requires_all_answer_keys():
    schema = answer_json_schema(["S1", "S2"])
    assert set(schema["required"]) == {"answer", "abstain", "conflict", "citations"}
    assert set(schema["$defs"]["LLMCitation"]["required"]) == {"source_id", "quote"}


def test_schema_restricts_source_id_to_this_requests_aliases():
    schema = answer_json_schema(["S1", "S2"])
    assert schema["$defs"]["LLMCitation"]["properties"]["source_id"]["enum"] == ["S1", "S2"]
