"""Output sanitising: nothing that could exfiltrate data through a rendered client reaches the response."""
import re

_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HTML = re.compile(r"<[^>\n]{1,200}>")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
# Characters a human reviewer cannot see but a model can read ("Unicode smuggling"): tag characters
# U+E0000-E007F, zero-width and joiner/word-joiner marks, bidi embeddings/overrides/isolates, BOM.
_INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤⁦-⁩﻿\U000e0000-\U000e007f]")


def strip_invisible(text: str) -> str:
    return _INVISIBLE.sub("", text)


def sanitize(text: str) -> str:
    text = strip_invisible(text)
    text = _IMAGE.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _HTML.sub("", text)
    text = _URL.sub("[link removed]", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()
