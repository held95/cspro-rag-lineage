"""Turns the LLM's claim into a verified response. The model is never trusted:

  * a cited source id must be one of THIS request's authorized aliases
  * a quote must appear verbatim (after normalisation) in THAT source
  * numbers in the answer must come from a verified quote (or the question)
  * the excerpt returned is server-extracted from the stored chunk, then sanitised
  * a "conflict" is only reported when two different documents are validly cited
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.modules.generation.normalize import find_quote
from app.modules.generation.sanitize import sanitize
from app.modules.retrieval.retriever import AuthorizedContext
from app.schemas.ask import ABSTAIN_TEXT, Citation, GroundedAnswer

_NUM = re.compile(r"\d+(?:[.,]\d+)?")


@dataclass
class VerifiedAnswer:
    status: str  # answered | abstained | conflict
    answer: str
    reason: str | None = None
    citations: list[Citation] = field(default_factory=list)
    conflict_doc_ids: list[str] = field(default_factory=list)


def _abstain(reason: str) -> VerifiedAnswer:
    return VerifiedAnswer("abstained", ABSTAIN_TEXT, reason)


def verify_answer(raw: GroundedAnswer, ctx: AuthorizedContext, question: str = "") -> VerifiedAnswer:
    if raw.abstain:
        return _abstain("insufficient_evidence")

    aliases = ctx.aliases
    citations: list[Citation] = []
    quotes: list[str] = []
    seen: set[tuple[str, int, int]] = set()
    for c in raw.citations:
        hit = aliases.get(c.source_id)
        if hit is None:
            continue  # unknown / foreign / fabricated id
        span = find_quote(hit.text, c.quote)
        if span is None or (c.source_id, *span) in seen:
            continue  # fabricated quote (or duplicate)
        seen.add((c.source_id, *span))
        excerpt = hit.text[span[0]:span[1]]
        quotes.append(excerpt)
        citations.append(Citation(
            source_id=c.source_id, doc_id=hit.doc_id, version=hit.version, chunk_id=hit.chunk_id,
            content_sha256=hit.content_sha256, source_uri=hit.source_uri, excerpt=sanitize(excerpt),
        ))
    if not citations:
        return _abstain("unverifiable_output")

    allowed_numbers = {n for text in [*quotes, question] for n in _NUM.findall(text)}
    if any(n not in allowed_numbers for n in _NUM.findall(raw.answer)):
        return _abstain("unverifiable_output")  # answer claims a figure no verified quote supports

    answer = sanitize(raw.answer)
    if not answer:
        return _abstain("unverifiable_output")

    docs = sorted({c.doc_id for c in citations})
    if raw.conflict and len(docs) >= 2:
        return VerifiedAnswer("conflict", answer, None, citations, docs)
    return VerifiedAnswer("answered", answer, None, citations)
