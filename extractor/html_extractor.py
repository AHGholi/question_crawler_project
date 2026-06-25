# extractor/html_extractor.py
from __future__ import annotations

import re
from html import unescape
from typing import List, Optional

from bs4 import BeautifulSoup, Tag

from utils.models import DocumentRecord, ExtractionResult


class HTMLExtractor:
    _WS_RE = re.compile(r"\s+")
    _SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

    # Non-content / structural noise tags
    _DROP_TAGS = {
        "script",
        "style",
        "noscript",
        "svg",
        "canvas",
        "iframe",
        "form",
        "button",
        "input",
        "select",
        "textarea",
        "nav",
        "header",
        "footer",
        "aside",
    }

    # Noisy hints (class/id/role/aria-label)
    _NOISY_HINTS_RE = re.compile(
        r"(menu|sidebar|cookie|consent|breadcrumb|social|share|subscribe|newsletter|"
        r"promo|advert|ads|banner|pagination|toolbar|related|recommend|comment|"
        r"login|signin|signup|language|locale|table-of-contents|toc|masthead|"
        r"header|footer)",
        flags=re.IGNORECASE,
    )

    # Keep this tight to avoid protecting noisy containers accidentally
    _PROTECTED_HINTS_RE = re.compile(
        r"(mw-content-text|bodyContent|entry-content|article[-_ ]?content|post[-_ ]?content)",
        flags=re.IGNORECASE,
    )

    # Site-specific noisy blocks (e.g., GeeksforGeeks)
    _SITE_NOISE_RE = re.compile(
        r"(gfg_header|headerMainList|containerSubheader|GFG_AD_|"
        r"sidebar|footer|comment|related|recommend)",
        flags=re.IGNORECASE,
    )

    def supports(self, document: DocumentRecord) -> bool:
        mt = (document.media_type or "").lower()
        path = str(document.path or document.source_path or "").lower()
        return "html" in mt or path.endswith(".html") or path.endswith(".htm")

    def extract(self, document: DocumentRecord) -> ExtractionResult:
        html = document.content or ""

        if not html:
            p = document.path or document.source_path
            if p is not None and p.exists():
                html = p.read_text(encoding=document.encoding or "utf-8", errors="ignore")

        soup = BeautifulSoup(html, "html.parser")
        self._remove_noise(soup)

        content_root = self._pick_content_root(soup)

        # primary
        raw_text = self._clean_text((content_root or soup).get_text(" ", strip=True))

        # fallback 1: body
        if not raw_text and soup.body is not None:
            raw_text = self._clean_text(soup.body.get_text(" ", strip=True))

        # fallback 2: full document
        if not raw_text:
            raw_text = self._clean_text(soup.get_text(" ", strip=True))

        clean_text = raw_text
        top_sentences = self._split_sentences(clean_text)[:80]
        summary = self._build_summary(top_sentences)

        return ExtractionResult(
            document=document,
            raw_text=raw_text,
            clean_text=clean_text,
            summary=summary,
            top_sentences=top_sentences,
            keywords=[],
            tokens=[],
            entities=[],
            keyword_scores={},
            errors=[],
        )

    def _remove_noise(self, soup: BeautifulSoup) -> None:
        # 1) Drop hard non-content tags
        for tag_name in self._DROP_TAGS:
            for node in soup.find_all(tag_name):
                node.decompose()

        # 2) Drop hidden blocks safely
        for node in soup.find_all(True):
            if not isinstance(node, Tag):
                continue
            attrs = node.attrs or {}
            style_val = attrs.get("style")
            style = str(style_val or "").replace(" ", "").lower()
            if "display:none" in style or "visibility:hidden" in style:
                node.decompose()

        # 3) Drop known site-noise blocks first (high precision)
        for node in soup.find_all(True):
            if not isinstance(node, Tag):
                continue

            attrs = node.attrs or {}
            raw_class = attrs.get("class")
            if isinstance(raw_class, list):
                class_text = " ".join(str(x) for x in raw_class if x is not None)
            else:
                class_text = str(raw_class or "")

            attrs_text = " ".join(
                [
                    class_text,
                    str(attrs.get("id") or ""),
                    str(attrs.get("role") or ""),
                    str(attrs.get("aria-label") or ""),
                ]
            ).strip()

            if not attrs_text:
                continue

            if self._SITE_NOISE_RE.search(attrs_text):
                node.decompose()

        # 4) Generic noisy hints with protection for real content containers
        for node in soup.find_all(True):
            if not isinstance(node, Tag):
                continue

            attrs = node.attrs or {}
            raw_class = attrs.get("class")
            if isinstance(raw_class, list):
                class_text = " ".join(str(x) for x in raw_class if x is not None)
            else:
                class_text = str(raw_class or "")

            attrs_text = " ".join(
                [
                    class_text,
                    str(attrs.get("id") or ""),
                    str(attrs.get("role") or ""),
                    str(attrs.get("aria-label") or ""),
                ]
            ).strip()

            if not attrs_text:
                continue

            if self._PROTECTED_HINTS_RE.search(attrs_text):
                continue

            if self._NOISY_HINTS_RE.search(attrs_text):
                node.decompose()

    def _pick_content_root(self, soup: BeautifulSoup) -> Optional[Tag]:
        # Put site-specific and semantic selectors first
        selectors = [
            ".MainArticleContent_articleMainContentCss__b_1_R",  # GfG
            ".article--viewer_content",                          # GfG
            "article",
            "main",
            "[role='main']",
            "#mw-content-text",   # wikipedia
            "#bodyContent",       # wikipedia
            "#content",
            ".entry-content",
            ".post-content",
            ".article-content",
            ".main-content",
            "#main-content",
            ".post",
            ".article",
            ".content",
        ]

        best: Optional[Tag] = None
        best_len = 0
        best_score = -1.0

        for sel in selectors:
            for node in soup.select(sel):
                if not isinstance(node, Tag):
                    continue

                txt = self._clean_text(node.get_text(" ", strip=True))
                if not txt:
                    continue

                txt_len = len(txt)
                alpha = sum(ch.isalpha() for ch in txt)
                alpha_ratio = alpha / max(len(txt), 1)

                # Prefer long, language-rich blocks
                score = txt_len * alpha_ratio

                if score > best_score or (score == best_score and txt_len > best_len):
                    best = node
                    best_len = txt_len
                    best_score = score

        if best is not None:
            return best

        return soup.body if soup.body is not None else soup

    def _split_sentences(self, text: str) -> List[str]:
        if not text:
            return []

        parts = self._SENT_SPLIT_RE.split(text)
        out: List[str] = []

        for p in parts:
            s = p.strip()
            if len(s) < 35:
                continue

            ls = s.lower()
            if s.count("/") > 6:
                continue
            if ls.startswith(("home ", "skip to ", "sign in", "log in")):
                continue
            if "privacy policy" in ls or "cookie policy" in ls:
                continue

            out.append(s)

        return out

    def _build_summary(self, sentences: List[str]) -> Optional[str]:
        if not sentences:
            return None
        return sentences[0][:320]

    @classmethod
    def _clean_text(cls, text: str) -> str:
        t = unescape(text or "")
        t = t.replace("\xa0", " ")
        t = cls._WS_RE.sub(" ", t).strip()
        return t