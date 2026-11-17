"""Output sanitising: nothing that could exfiltrate data through a rendered client reaches the response."""
import re

_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HTML = re.compile(r"<[^>\n]{1,200}>")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)


def sanitize(text: str) -> str:
    text = _IMAGE.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _HTML.sub("", text)
    text = _URL.sub("[link removed]", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()
