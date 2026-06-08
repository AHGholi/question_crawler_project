from __future__ import annotations

import pytest

pytest.importorskip("bs4")

from extractor.cleaner import clean_text, collapse_whitespace, strip_html_tags


def test_strip_html_tags_removes_all_except_keep():
    html = "<div><p>hello <strong>world</strong></p></div>"
    assert strip_html_tags(html) == "hello world"
    preserved = strip_html_tags(html, keep_tags=("strong",))
    assert preserved == "hello <strong>world</strong>"


def test_collapse_whitespace_normalizes_spaces():
    text = "Line 1\t \nLine 2   Line 3"
    assert collapse_whitespace(text) == "Line 1 Line 2 Line 3"


def test_clean_text_full_pipeline():
    html_snippet = "<p>Example\u000btext&nbsp;</p>"
    cleaned = clean_text(html_snippet, strip_html=True, lowercase=True)
    assert cleaned == "example text"
