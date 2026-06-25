from __future__ import annotations

import re
from typing import List

# Keep your existing spaCy wiring if you already have it in this file.
# This module-level cache avoids reloading the model repeatedly.
_NLP = None


def _get_nlp():
    """
    Returns a spaCy pipeline with sentencizer/parser enabled.
    Falls back gracefully if spaCy/model is unavailable.
    """
    global _NLP
    if _NLP is not None:
        return _NLP

    try:
        import spacy

        # Try common English model first
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            # Fallback to blank English with sentencizer
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")

        # Ensure sentence boundaries exist
        if "parser" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

        _NLP = nlp
        return _NLP
    except Exception:
        _NLP = False
        return _NLP


def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def approx_token_count(text: str) -> int:
    """
    Lightweight token estimate (word-ish units).
    Good enough for chunk sizing without a tokenizer dependency.
    """
    if not text:
        return 0
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def split_into_sentences(text: str) -> List[str]:
    """
    Sentence split with spaCy when available; regex fallback otherwise.
    """
    text = normalize_whitespace(text)
    if not text:
        return []

    nlp = _get_nlp()
    if nlp:
        try:
            doc = nlp(text)
            sents = [s.text.strip() for s in doc.sents if s.text and s.text.strip()]
            if sents:
                return sents
        except Exception:
            pass

    # Regex fallback (not perfect, but robust)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+", text)
    sents = [p.strip() for p in parts if p and p.strip()]
    return sents


def chunk_text(
    text: str,
    max_tokens: int = 220,
    overlap_sentences: int = 1,
    min_chunk_chars: int = 180,
) -> List[str]:
    """
    Build coherent overlapping chunks from sentences.

    Args:
        text: input text
        max_tokens: approximate max tokens per chunk
        overlap_sentences: number of trailing sentences to carry into next chunk
        min_chunk_chars: drop chunks shorter than this unless it's the only chunk

    Returns:
        List of chunk strings.
    """
    text = normalize_whitespace(text)
    if not text:
        return []

    sentences = split_into_sentences(text)
    if not sentences:
        return [text] if len(text) >= min_chunk_chars else ([text] if text else [])

    chunks: List[str] = []
    i = 0
    n = len(sentences)

    while i < n:
        current: List[str] = []
        cur_tokens = 0
        j = i

        while j < n:
            s = sentences[j].strip()
            if not s:
                j += 1
                continue

            s_tokens = approx_token_count(s)

            # If first sentence itself is too big, force-include it
            if not current and s_tokens >= max_tokens:
                current.append(s)
                j += 1
                break

            # If adding sentence exceeds cap, stop chunk growth
            if current and (cur_tokens + s_tokens > max_tokens):
                break

            current.append(s)
            cur_tokens += s_tokens
            j += 1

        if not current:
            # safety
            i += 1
            continue

        chunk = " ".join(current).strip()
        if chunk:
            chunks.append(chunk)

        # Advance with overlap
        if j >= n:
            break

        # Rewind start by overlap_sentences from j
        if overlap_sentences > 0:
            i = max(i + 1, j - overlap_sentences)
        else:
            i = j

    # Remove tiny chunks (except if it's the only one)
    if len(chunks) > 1:
        chunks = [c for c in chunks if len(c) >= min_chunk_chars] or chunks

    # De-duplicate exact repeats while preserving order
    deduped: List[str] = []
    seen = set()
    for c in chunks:
        key = c.strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)

    return deduped
