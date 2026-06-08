from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from utils.models import ExtractionResult, QuestionItem

from question_generator_old.configuration import GeneratorConfig

_ENUMERATION_REGEXES = [
    re.compile(r"(?:include|includes|including)\s+(?P<listing>[^.?!;]+)", re.IGNORECASE),
    re.compile(r"(?:consist of|consists of)\s+(?P<listing>[^.?!;]+)", re.IGNORECASE),
    re.compile(r"(?:comprise|comprises)\s+(?P<listing>[^.?!;]+)", re.IGNORECASE),
    re.compile(r"(?:feature|features)\s+(?P<listing>[^.?!;]+)", re.IGNORECASE),
    re.compile(r"(?:contain|contains)\s+(?P<listing>[^.?!;]+)", re.IGNORECASE),
    re.compile(r"(?:offer|offers)\s+(?P<listing>[^.?!;]+)", re.IGNORECASE),
    re.compile(r"(?:provide|provides)\s+(?P<listing>[^.?!;]+)", re.IGNORECASE),
    re.compile(r"(?:such as|like)\s+(?P<listing>[^.?!;]+)", re.IGNORECASE),
]


def generate_listing_questions(
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

    scores = [s for _, s in ranked_sentences]
    min_s = min(scores)
    max_s = max(scores)

    # 1) Keyword-anchored (current behavior)
    if keywords:
        for keyword in keywords:
            keyword_lower = keyword.lower()
            prompt = f"List the key facts about {keyword} mentioned."

            if prompt.lower() in seen_prompts:
                continue

            for sentence, raw_score in ranked_sentences:
                if keyword_lower not in sentence.lower():
                    continue

                enumeration = _extract_enumeration_items(sentence, keyword_lower)
                if not enumeration:
                    continue

                answer = _format_enumeration_answer(enumeration, config.answer_max_chars)
                confidence = _compute_confidence(_normalize_score(raw_score, min_s, max_s), config.confidence_decay)

                items.append(
                    QuestionItem(
                        question=prompt,
                        answer=answer,
                        source_sentence=sentence,
                        confidence=confidence,
                        tags=["listing", f"keyword:{keyword_lower}"],
                        source_document_id=_get_document_id(extraction),
                    )
                )
                seen_prompts.add(prompt.lower())
                break

            if len(items) >= config.max_questions:
                return items

    # 2) Sentence-first fallback
    for sentence, raw_score in ranked_sentences:
        norm_score = _normalize_score(raw_score, min_s, max_s)
        if norm_score < config.min_sentence_score:
            continue

        enumeration = _extract_enumeration_items(sentence, keyword_lower="")
        if not enumeration:
            continue

        subject = _extract_listing_subject(sentence)
        if subject:
            prompt = f"What items are included in {subject}?"
        else:
            prompt = "What items are listed?"

        if prompt.lower() in seen_prompts:
            continue

        answer = _format_enumeration_answer(enumeration, config.answer_max_chars)
        confidence = _compute_confidence(norm_score, config.confidence_decay)

        items.append(
            QuestionItem(
                question=prompt,
                answer=answer,
                source_sentence=sentence,
                confidence=confidence,
                tags=["listing", "sentence-first"],
                source_document_id=_get_document_id(extraction),
            )
        )

        seen_prompts.add(prompt.lower())
        if len(items) >= config.max_questions:
            break

    return items


def _extract_enumeration_items(sentence: str, keyword_lower: str) -> List[str] | None:
    lowered = sentence.lower()
    idx = lowered.find(keyword_lower) if keyword_lower else -1
    search_region = sentence if idx == -1 else sentence[idx:]

    for regex in _ENUMERATION_REGEXES:
        match = regex.search(search_region)
        if not match:
            continue
        list_text = match.group("listing").strip()
        items = _split_enumeration(list_text)
        if len(items) >= 2:
            return items

    fallback_items = _split_enumeration(search_region)
    return fallback_items if len(fallback_items) >= 2 else None


def _split_enumeration(text: str) -> List[str]:
    if not text:
        return []

    cleaned = text.strip(" :;,.")
    cleaned = re.split(r"\b(?:that|which)\b", cleaned, maxsplit=1)[0]
    cleaned = re.sub(r"\s+(?:and|or)\s+", ", ", cleaned)

    parts = [part.strip(" :;,.") for part in cleaned.split(",")]
    deduped: List[str] = []
    seen: set[str] = set()

    for part in parts:
        if len(part) < 2:
            continue
        normalized = part.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(part)

    return deduped


def _extract_listing_subject(sentence: str) -> str | None:
    """
    Heuristic: take the phrase before the enumeration trigger.
    Example: "The program includes X, Y, Z" -> "The program"
    """
    for regex in _ENUMERATION_REGEXES:
        match = regex.search(sentence)
        if not match:
            continue
        prefix = sentence[: match.start()].strip(" ,:;-")
        if 1 <= len(prefix.split()) <= 6:
            return prefix
    return None


def _format_enumeration_answer(items: List[str], max_chars: int) -> str:
    joined = "; ".join(items)
    if len(joined) <= max_chars:
        return joined

    truncated = joined[: max_chars - 1].rstrip()
    if "; " in truncated:
        truncated = truncated.rsplit("; ", 1)[0]
    elif " " in truncated:
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
