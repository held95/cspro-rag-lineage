"""Text normalisation shared by chunking, quote verification and the fine-tune scorer.

One normaliser everywhere means "verbatim quote" has a single, testable meaning:
NFKC, zero-width/format characters removed, whitespace collapsed to one space,
case-sensitive. ``index_map`` maps every normalised char back to its position in
the original text so the excerpt returned to clients is the ORIGINAL span.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedText:
    text: str
    index_map: tuple[int, ...]  # index_map[i] = original index of normalised char i


def normalize(text: str) -> NormalizedText:
    out: list[str] = []
    idx_map: list[int] = []
    for idx, ch in enumerate(text):
        for c in unicodedata.normalize("NFKC", ch):
            if c.isspace():
                if out and out[-1] == " ":
                    continue
                out.append(" ")
                idx_map.append(idx)
                continue
            category = unicodedata.category(c)
            if category in ("Cf", "Cc"):  # zero-width, bidi marks, control chars
                continue
            out.append(c)
            idx_map.append(idx)
    return NormalizedText("".join(out), tuple(idx_map))


def find_quote(chunk: str, quote: str) -> tuple[int, int] | None:
    """Return the (start, end) span in the ORIGINAL chunk if ``quote`` appears verbatim
    after normalisation, else None. Empty/blank quotes never match."""
    nq = normalize(quote).text.strip()
    if not nq:
        return None
    nc = normalize(chunk)
    pos = nc.text.find(nq)
    if pos == -1:
        return None
    start = nc.index_map[pos]
    end = nc.index_map[pos + len(nq) - 1] + 1
    return start, end
