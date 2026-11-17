from __future__ import annotations

import logging

import httpx
from pydantic import ValidationError

from app.core.errors import AppError
from app.modules.generation.llm import LLMClient
from app.modules.generation.prompt import PromptSource, build_messages, new_nonce
from app.modules.generation.verify import verify_answer
from app.modules.retrieval.retriever import Retriever
from app.modules.tenancy.principal import Principal
from app.schemas.ask import (ABSTAIN_TEXT, AskRequest, AskResponse, ConflictInfo, GroundedAnswer,
                             answer_json_schema)

log = logging.getLogger("app.ask")


class AskService:
    def __init__(self, retriever: Retriever, llm: LLMClient, model_name: str):
        self.retriever, self.llm, self.model_name = retriever, llm, model_name

    def ask(self, principal: Principal, req: AskRequest, request_id: str) -> AskResponse:
        ctx = self.retriever.retrieve(principal, req.question, req.top_k)  # tenant + ACL enforced here
        if not ctx.hits:
            return self._response("abstained", ABSTAIN_TEXT, request_id, reason="insufficient_evidence")

        aliases = ctx.aliases
        sources = [PromptSource(a, h.kind, h.title, h.text) for a, h in aliases.items()]
        messages = build_messages(req.question, sources, new_nonce(), conflict_hint=bool(ctx.conflict_doc_ids))
        schema = answer_json_schema(list(aliases))

        parsed = self._generate_parsed(messages, schema)
        if parsed is None:
            return self._response("abstained", ABSTAIN_TEXT, request_id, reason="unverifiable_output")

        v = verify_answer(parsed, ctx, req.question)
        return self._response(v.status, v.answer, request_id, reason=v.reason, citations=v.citations,
                              conflict_docs=v.conflict_doc_ids)

    # ---- helpers -----------------------------------------------------------------------------
    def _generate_parsed(self, messages, schema) -> GroundedAnswer | None:
        for attempt in (1, 2):  # one retry when the model returns malformed JSON
            try:
                raw = self.llm.generate(messages, schema)
            except httpx.TimeoutException as exc:
                raise AppError("LLM_TIMEOUT", 504, "The language model timed out") from exc
            except (httpx.HTTPError, ConnectionError, OSError) as exc:
                raise AppError("LLM_UNAVAILABLE", 503, "The language model is unavailable") from exc
            try:
                return GroundedAnswer.model_validate_json(raw)
            except ValidationError:
                log.warning("model returned invalid JSON (attempt %s)", attempt)
        return None

    def _response(self, status, answer, request_id, reason=None, citations=None, conflict_docs=None) -> AskResponse:
        return AskResponse(
            status=status, answer=answer, reason=reason, citations=citations or [],
            conflict=ConflictInfo(detected=status == "conflict", doc_ids=conflict_docs or []),
            model=self.model_name, request_id=request_id,
        )
