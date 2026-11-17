import pytest

from app.modules.generation.normalize import find_quote, normalize

pytestmark = pytest.mark.unit

CHUNK = "Refunds are accepted within 30 days of purchase.\nContact support for details."


def test_exact_quote_is_found_and_excerpt_is_original_text():
    span = find_quote(CHUNK, "within 30 days of purchase")
    assert span is not None
    assert CHUNK[span[0]:span[1]] == "within 30 days of purchase"


def test_whitespace_and_nbsp_differences_still_match():
    chunk = "Refunds   are accepted\nwithin 30 days."
    span = find_quote(chunk, "are accepted within 30 days")
    assert span is not None
    assert "30 days" in chunk[span[0]:span[1]]


def test_zero_width_characters_are_ignored_in_chunk_and_quote():
    chunk = "Refund​ window is 30 days"
    assert find_quote(chunk, "Refund window is 30 days") is not None
    assert find_quote("Refund window is 30 days", "Refu​nd window") is not None


def test_case_change_fails():
    assert find_quote(CHUNK, "WITHIN 30 DAYS") is None


def test_homoglyph_substitution_fails():
    forged = "within 30 dаys"  # Cyrillic a
    assert find_quote(CHUNK, forged) is None


def test_fabricated_quote_fails_and_empty_quote_fails():
    assert find_quote(CHUNK, "refunds are accepted within 90 days") is None
    assert find_quote(CHUNK, "   ") is None


def test_normalize_returns_index_map_same_length_as_text():
    n = normalize("a  b​c")
    assert n.text == "a bc"
    assert len(n.index_map) == len(n.text)
