#extractor/cleaner.py
from __future__ import annotations

import html
import re
from typing import Any, Iterable, Optional, Sequence, Set, TYPE_CHECKING


try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    from bs4 import BeautifulSoup as BeautifulSoupType
else:
    BeautifulSoupType = Any

CONTROL_CHARS = "".join(map(chr, range(0, 32))) + "".join(map(chr, range(127, 160)))
CONTROL_CHAR_RE = re.compile(f"[{re.escape(CONTROL_CHARS)}]")

WHITESPACE_RE = re.compile(r"\s+")


def remove_control_characters(text: str) -> str:
    return CONTROL_CHAR_RE.sub(" ", text)


def collapse_whitespace(text: str, collapse_to: str = " ") -> str:
    if not text:
        return ""
    return WHITESPACE_RE.sub(collapse_to, text).strip()


def _render_contents(soup: BeautifulSoupType) -> str:
    """Return the string representation of the soup without <html>/<body> wrappers."""
    container = soup.body or soup
    return container.decode_contents(formatter="html").strip()


def strip_html_tags(text: str, keep_tags: Optional[Sequence[str]] = None) -> str:
    if not text:
        return ""

    keep: Set[str] = {tag.lower() for tag in keep_tags or ()}

    if BeautifulSoup is not None:
        soup = BeautifulSoup(text, "html.parser")
        if keep:
            for tag in soup.find_all(True):
                tag_name = tag.name.lower()
                if tag_name in keep:
                    # Drop attributes for cleanliness.
                    tag.attrs = {}
                else:
                    tag.unwrap()
            return _render_contents(soup)
        return soup.get_text(separator=" ", strip=True)

    # Fallback: naive tag removal (keep-tags unsupported without BeautifulSoup).
    return re.sub(r"<[^>]+>", "", text)


def clean_text(
    text: str,
    *,
    strip_html: bool = True,
    keep_tags: Optional[Sequence[str]] = None,
    lowercase: bool = False,
    collapse_to: str = " ",
) -> str:
    if not text:
        return ""

    working = text

    if strip_html:
        working = strip_html_tags(working, keep_tags=keep_tags)

    # Decode entities and normalise non-breaking spaces before collapsing.
    working = html.unescape(working).replace("\u00a0", " ")

    working = remove_control_characters(working)
    working = collapse_whitespace(working, collapse_to=collapse_to)

    if lowercase:
        working = working.lower()

    return working.strip()


__all__ = [
    "remove_control_characters",
    "collapse_whitespace",
    "strip_html_tags",
    "clean_text",
]
