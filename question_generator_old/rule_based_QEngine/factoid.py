from __future__ import annotations

import re
from typing import List, Tuple

from utils.models import ExtractionResult, QuestionItem

from ..configuration import GeneratorConfig


_LOW_VALUE_TOKENS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "onto",
    "or",
    "over",
    "the",
    "through",
    "to",
    "toward",
    "towards",
    "with",
    "within",
}


def generate_factoid_questions(
    *,
    extraction: ExtractionResult,
    ranked_sentences: List[Tuple[str, float]],
    config: GeneratorConfig,
) -> List[QuestionItem]:
    """
    Sentence-first question generation:
      1) Use definitional/appositive patterns when possible.
      2) Use keyword-anchored questions as an enhancement.
      3) Use fallback subject extraction if no keyword exists.
    """
    keywords = list(extraction.keywords or [])
    items: List[QuestionItem] = []

    if not ranked_sentences:
        return []

    # Normalize sentence scores
    scores = [s for _, s in ranked_sentences]
    min_s = min(scores)
    max_s = max(scores)

    for rank_idx, (sentence, raw_score) in enumerate(ranked_sentences):
        norm_score = _normalize_score(raw_score, min_s, max_s)
        if norm_score < config.min_sentence_score:
            continue

        sentence = sentence.strip()
        if not sentence:
            continue

        # 1) Definition / appositive patterns first
        definition = _extract_definition(sentence)
        if definition:
            subject, predicate = definition
            if _is_valid_keyword(subject):
                q_text = f"What is {subject}?"
                answer_text, extracted = _ensure_answer_quality(
                    answer_text=predicate,
                    extracted=True,
                    sentence=sentence,
                )
                confidence = _compute_confidence(
                    score=norm_score,
                    rank_index=rank_idx,
                    decay=config.confidence_decay,
                    extracted=True,
                )
                items.append(
                    QuestionItem(
                        question=q_text,
                        answer=answer_text,
                        source_document_id=extraction.document_id,
                        source_sentence=sentence,
                        confidence=confidence,
                        tags=["factoid", "sentence-first", "pattern:definition"],
                    )
                )
                if len(items) >= config.max_questions:
                    return items
                continue

        # 2) Keyword-anchored questions (if keyword appears)
        present_keywords = _keywords_in_sentence(keywords, sentence)
        if present_keywords:
            present_keywords = present_keywords[: config.max_keywords_per_question]

            for keyword in present_keywords:
                if not _is_valid_keyword(keyword):
                    continue

                question_text = _build_factoid_question(keyword)
                answer_text, extracted = _extract_answer_span(keyword, sentence)
                answer_text, extracted = _ensure_answer_quality(
                    answer_text=answer_text,
                    extracted=extracted,
                    sentence=sentence,
                )

                if len(answer_text) > config.answer_max_chars:
                    answer_text = answer_text[: config.answer_max_chars - 1].rstrip() + "…"

                confidence = _compute_confidence(
                    score=norm_score,
                    rank_index=rank_idx,
                    decay=config.confidence_decay,
                    extracted=extracted,
                )

                items.append(
                    QuestionItem(
                        question=question_text,
                        answer=answer_text,
                        source_document_id=extraction.document_id,
                        source_sentence=sentence,
                        confidence=confidence,
                        tags=["factoid", "sentence-first", f"keyword:{keyword.lower()}"],
                    )
                )

                if len(items) >= config.max_questions:
                    return items

            continue

        # 3) Fallback: subject-based question from sentence
        subject = _extract_subject(sentence)
        if subject and _is_valid_keyword(subject):
            q_text = f"What does the text say about {subject}?"
            answer_text, extracted = _ensure_answer_quality(
                answer_text=None,
                extracted=False,
                sentence=sentence,
            )
            confidence = _compute_confidence(
                score=norm_score,
                rank_index=rank_idx,
                decay=config.confidence_decay,
                extracted=False,
            )
            items.append(
                QuestionItem(
                    question=q_text,
                    answer=answer_text,
                    source_document_id=extraction.document_id,
                    source_sentence=sentence,
                    confidence=confidence,
                    tags=["factoid", "sentence-first", "pattern:fallback"],
                )
            )

            if len(items) >= config.max_questions:
                return items

    return items


def _normalize_score(score: float, min_s: float, max_s: float) -> float:
    try:
        s = float(score)
    except (TypeError, ValueError):
        return 0.0

    if max_s == min_s:
        return max(0.0, min(1.0, s))

    norm = (s - min_s) / (max_s - min_s)
    return max(0.0, min(1.0, norm))


def _keywords_in_sentence(keywords: List[str], sentence: str) -> List[str]:
    sentence_norm = " " + sentence.lower() + " "
    matched: List[str] = []

    for keyword in keywords:
        keyword_norm = keyword.strip().lower()
        if not keyword_norm:
            continue
        pattern = r"\b" + re.escape(keyword_norm) + r"\b"
        if re.search(pattern, sentence_norm) or keyword_norm in sentence_norm:
            matched.append(keyword)

    unique: List[str] = []
    seen = set()
    for keyword in matched:
        lowered = keyword.lower()
        if lowered not in seen:
            seen.add(lowered)
            unique.append(keyword)
    return unique


def _is_valid_keyword(keyword: str) -> bool:
    tokens = [tok for tok in re.split(r"\s+", keyword.strip()) if tok]
    if not tokens:
        return False

    if tokens[0].lower() in _LOW_VALUE_TOKENS or tokens[-1].lower() in _LOW_VALUE_TOKENS:
        return False

    if len(tokens) == 1 and len(tokens[0]) <= 3:
        return False

    if all(len(tok) <= 2 for tok in tokens):
        return False

    return True


def _build_factoid_question(keyword: str) -> str:
    term = keyword.strip()
    return f"What is {term}?" if term else "What is it?"


def _extract_definition(sentence: str) -> tuple[str, str] | None:
    """
    Extract (subject, predicate) from definition-like patterns.
    Examples:
      - X is Y
      - X refers to Y
      - X is defined as Y
      - X, a Y, ...
    """
    s = sentence.strip()

    # Pattern: "X is/are/was/were Y"
    m = re.search(
        r"^(?P<subj>.+?)\s+(is|are|was|were|means|refers to|is defined as)\s+(?P<pred>.+?)(?:[.;:()\[\]–—-]|$)",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        subj = m.group("subj").strip(" -:;,.()[]\"'")
        pred = m.group("pred").strip(" -:;,.()[]\"'")
        if subj and pred:
            return subj, pred

    # Pattern: "X, a/an Y, ..."
    m = re.search(
        r"^(?P<subj>.+?),\s+(an?|the)\s+(?P<pred>.+?),(?:\s|$)",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        subj = m.group("subj").strip(" -:;,.()[]\"'")
        pred = m.group("pred").strip(" -:;,.()[]\"'")
        if subj and pred:
            return subj, pred

    return None


def _extract_subject(sentence: str) -> str | None:
    """
    Heuristic subject extraction: take the lead phrase before a common verb.
    """
    s = sentence.strip()
    m = re.search(
        r"^(?P<subj>.+?)\s+(is|are|was|were|has|have|does|do|can|will|provides|enables|allows|supports)\b",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        subj = m.group("subj").strip(" -:;,.()[]\"'")
        # Avoid very long / noisy subjects
        if 1 <= len(subj.split()) <= 6:
            return subj

    # Fallback: first 1-4 tokens if they look meaningful
    tokens = [t for t in re.split(r"\s+", s) if t]
    if 1 <= len(tokens) <= 4:
        return " ".join(tokens)
    if len(tokens) >= 4:
        candidate = " ".join(tokens[:4]).strip(" -:;,.()[]\"'")
        return candidate if candidate else None

    return None


def _extract_answer_span(keyword: str, sentence: str) -> tuple[str | None, bool]:
    s = sentence.strip()
    k = re.escape(keyword.strip())

    patterns = [
        rf"\b{k}\b\s+(?:is|are|was|were)\s+(?:an?\s+|the\s+)?(.+?)(?:[.;:()\[\]–—-]|$)",
        rf"(?:^|[\s,;:()\[\]–—-])(.+?)\s+(?:is|are|was|were)\s+(?:an?\s+|the\s+)?\b{k}\b(?:[.;:()\[\]–—-]|$)",
        rf"\b{k}\b,\s+(?:an?\s+|the\s+)?(.+?),(?:\s|$)",
        rf"\b{k}\b\s*\((.+?)\)",
        rf"\b{k}\b\s*[:–—-]\s*(.+?)(?:[.;]|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, s, flags=re.IGNORECASE)
        if match:
            answer = match.group(1).strip(" -:;,.()[]")
            if answer:
                return answer, True

    if re.search(rf"\b{k}\b", s, flags=re.IGNORECASE):
        candidate = re.sub(rf"\b{k}\b", "", s, flags=re.IGNORECASE).strip()
        candidate = re.sub(r"^(is|are|was|were|,|:|–|—|-)\s*", "", candidate, flags=re.IGNORECASE)
        candidate = candidate.strip(" -:;,.()[]")
        if candidate:
            return candidate, False

    return None, False


def _ensure_answer_quality(
    *,
    answer_text: str | None,
    extracted: bool,
    sentence: str,
) -> tuple[str, bool]:
    if answer_text:
        cleaned = answer_text.strip(" \t\r\n-:;,.()[]\"'")
        token_count = len(cleaned.split())
        if token_count >= 3 and any(ch.isalpha() for ch in cleaned):
            return cleaned, extracted

    cleaned_sentence = sentence.strip()
    return cleaned_sentence, False


def _compute_confidence(*, score: float, rank_index: int, decay: float, extracted: bool) -> float:
    base = max(0.0, min(1.0, score))
    decayed = base * max(0.0, 1.0 - decay * rank_index)
    if extracted:
        decayed = min(1.0, decayed + 0.05)
    return max(0.0, min(1.0, decayed))
