from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from utils.models import ExtractionResult, QuestionItem

from question_generator_old.configuration import GeneratorConfig


_DEFINITION_PATTERNS = [
    r"(?P<subject>.+?)\s+is\s+(?:an?\s+|the\s+)?(?P<definition>[^.?!;]+)",
    r"(?P<subject>.+?)\s+are\s+(?:an?\s+|the\s+)?(?P<definition>[^.?!;]+)",
    r"(?P<subject>.+?)\s+refers\s+to\s+(?P<definition>[^.?!;]+)",
    r"(?P<subject>.+?)\s+means\s+(?P<definition>[^.?!;]+)",
    r"(?P<subject>.+?)\s+represents\s+(?P<definition>[^.?!;]+)",
]

_APPOSITIVE_PATTERN = re.compile(
    r"^(?P<subject>.+?),\s+(?:an?|the)\s+(?P<definition>.+?),(?:\s|$)",
    flags=re.IGNORECASE,
)


def generate_definition_questions(
    *,
    extraction: ExtractionResult,
    ranked_sentences: Sequence[tuple[str, float]],
    config: GeneratorConfig,
) -> List[QuestionItem]:
    keywords = [kw.strip() for kw in (extraction.keywords or []) if kw and kw.strip()]
    items: List[QuestionItem] = []
    seen_prompts: set[str] = set()

    if not ranked_sentences:
        return []

    # Normalize sentence scores for filtering
    scores = [s for _, s in ranked_sentences]
    min_s = min(scores)
    max_s = max(scores)

    # 1) Keyword-anchored definitions (current behavior)
    if keywords:
        for keyword in keywords:
            prompt = f"Define {keyword}."
            if prompt.lower() in seen_prompts:
                continue

            keyword_lower = keyword.lower()
            for sentence, raw_score in ranked_sentences:
                if keyword_lower not in sentence.lower():
                    continue

                definition = _extract_definition_clause(sentence, keyword_lower)
                if not definition:
                    continue

                answer = _truncate_definition(definition, config.answer_max_chars)
                confidence = _compute_confidence(_normalize_score(raw_score, min_s, max_s), config.confidence_decay)

                items.append(
                    QuestionItem(
                        question=prompt,
                        answer=answer,
                        source_sentence=sentence,
                        confidence=confidence,
                        tags=["definition", f"keyword:{keyword_lower}"],
                        source_document_id=_get_document_id(extraction),
                    )
                )

                seen_prompts.add(prompt.lower())
                break

            if len(items) >= config.max_questions:
                return items

    # 2) Sentence-first fallback (no keywords or no matches)
    for sentence, raw_score in ranked_sentences:
        norm_score = _normalize_score(raw_score, min_s, max_s)
        if norm_score < config.min_sentence_score:
            continue

        subject, definition = _extract_definition_pair(sentence)
        if not subject or not definition:
            continue

        prompt = f"Define {subject}."
        if prompt.lower() in seen_prompts:
            continue

        answer = _truncate_definition(definition, config.answer_max_chars)
        confidence = _compute_confidence(norm_score, config.confidence_decay)

        items.append(
            QuestionItem(
                question=prompt,
                answer=answer,
                source_sentence=sentence,
                confidence=confidence,
                tags=["definition", "sentence-first"],
                source_document_id=_get_document_id(extraction),
            )
        )

        seen_prompts.add(prompt.lower())
        if len(items) >= config.max_questions:
            break

    return items


def _extract_definition_pair(sentence: str) -> Tuple[str | None, str | None]:
    s = sentence.strip()

    for pattern in _DEFINITION_PATTERNS:
        compiled = re.compile(pattern, flags=re.IGNORECASE)
        match = compiled.search(s)
        if match:
            subject = match.group("subject").strip(" ,:;-")
            definition = match.group("definition").strip(" ,:;-")
            if subject and definition:
                return subject, definition

    # Appositive pattern: "X, a Y,"
    match = _APPOSITIVE_PATTERN.search(s)
    if match:
        subject = match.group("subject").strip(" ,:;-")
        definition = match.group("definition").strip(" ,:;-")
        if subject and definition:
            return subject, definition

    return None, None


def _extract_definition_clause(sentence: str, keyword_lower: str) -> str | None:
    lowered_sentence = sentence.lower()
    keyword_pattern = re.escape(keyword_lower)

    for pattern in _DEFINITION_PATTERNS:
        compiled = re.compile(pattern.format(kw=keyword_pattern), flags=re.IGNORECASE)
        match = compiled.search(sentence)
        if match:
            clause = match.group("definition")
            cleaned = clause.strip(" ,:;-")
            if cleaned:
                return cleaned

    idx = lowered_sentence.find(keyword_lower)
    if idx == -1:
        return None

    remainder = sentence[idx + len(keyword_lower) :]
    cleaned = remainder.strip(" ,:;-")
    return cleaned or None


def _truncate_definition(definition: str, max_chars: int) -> str:
    definition = definition.strip()
    if len(definition) <= max_chars:
        return definition

    truncated = definition[: max_chars - 1].rstrip()
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return truncated + "…"


def _compute_confidence(normalized_score: float, decay: float) -> float:
    confidence = max(0.05, min(1.0, normalized_score - decay))
    return round(confidence, 3)


def _normalize_score(score: float | None, min_s: float, max_s: float) -> float:
    if score is None:
        return 0.5

    try:
        s = float(score)
    except (TypeError, ValueError):
        return 0.0

    if max_s == min_s:
        return max(0.0, min(1.0, s))

    return max(0.0, min(1.0, (s - min_s) / (max_s - min_s)))


def _get_document_id(extraction: ExtractionResult) -> str:
    if hasattr(extraction, "document_id") and extraction.document_id:
        return str(extraction.document_id)
    if hasattr(extraction, "document") and hasattr(extraction.document, "id") and extraction.document.id:
        return str(extraction.document.id)
    return ""
