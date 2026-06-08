# tests/crawler/test_url_filter.py
from crawler.url_filter import normalize_url, candidate_allowed

def test_normalize_url_strips_fragment():
    url = "https://example.com/path#section"
    assert normalize_url(url) == "https://example.com/path"

def test_candidate_allowed_blocks_disallowed_extension():
    url = "https://example.com/file.zip"
    assert candidate_allowed(url) is False

def test_candidate_allowed_respects_allow_domains():
    url = "https://docs.python.org/tutorial"
    assert candidate_allowed(url, allow_domains=["python.org"]) is True
    assert candidate_allowed(url, allow_domains=["example.com"]) is False
