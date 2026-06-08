# tests/integration/test_phase2_processing_pipeline.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from utils.models import DocumentRecord
from processor.pipeline import ProcessorPipeline
from processor.text_processing import TextProcessor
from processor.vectorizer import TextVectorizer
from processor.similarity import SimilarityEngine

from extractor.html_extractor import HTMLExtractor


def _parse_timestamp_prefix(name: str) -> str | None:
    match = re.match(r"^(\d{8}-\d{6})_", name)
    return match.group(1) if match else None


def _domain_tag(name: str) -> str:
    lowered = name.lower()
    if "wikipedia" in lowered:
        return "wikipedia"
    if "ibm" in lowered:
        return "ibm"
    if "developers-google" in lowered or "google" in lowered:
        return "google"
    if "mitsloan" in lowered or "mit-sloan" in lowered or "mit" in lowered:
        return "mitsloan"
    if "coursera" in lowered:
        return "coursera"
    return "other"


def _expected_token_hints(tag: str) -> List[str]:
    mapping = {
        "ibm": ["ibm", "supervise", "unsupervise", "reinforcement", "algorithm", "model", "data", "ai"],
        "wikipedia": ["statistic", "algorithm", "model", "train", "classification", "supervise", "unsupervise", "reinforcement"],
        "google": ["google", "developer", "tensorflow", "vertex", "model", "dataset", "tutorial"],
        "mitsloan": ["mit", "sloan", "business", "predictive", "automation", "bias", "application"],
        "coursera": ["coursera", "course", "supervise", "regression", "classification", "andrew", "ng", "gradient"],
    }
    return mapping.get(tag, ["machine", "learn"])


def _split_sentences(text: str, processor: TextProcessor) -> List[str]:
    doc = processor.nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def _rank_sentences_with_fallback(clean_text: str, query: str, processor: TextProcessor) -> List[Tuple[str, float]]:
    sentences = _split_sentences(clean_text, processor)
    if not sentences:
        return []

    try:
        from extractor.sentence_ranker import SentenceRanker  # type: ignore

        try:
            ranker = SentenceRanker()
        except TypeError:
            ranker = SentenceRanker()

        candidate_calls = [
            lambda: ranker.rank_sentences(query, sentences, top_k=5),
            lambda: ranker.rank_sentences(query, sentences),
        ]

        for call in candidate_calls:
            try:
                results = call()
                if not results:
                    continue

                normalized: List[Tuple[str, float]] = []
                for item in results:
                    if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[0], str):
                        normalized.append((item[0], float(item[1])))
                    elif isinstance(item, str):
                        normalized.append((item, 1.0))
                    else:
                        text = getattr(item, "text", None)
                        score = getattr(item, "score", None)
                        if isinstance(text, str) and score is not None:
                            normalized.append((text, float(score)))

                if normalized:
                    return normalized[:5]
            except TypeError:
                continue
            except Exception:
                break
    except Exception:
        pass

    processor_tokens = [processor.process(sentence) for sentence in sentences]
    vectorizer = TextVectorizer()
    sentence_matrix = vectorizer.fit_transform(processor_tokens)
    query_tokens = processor.process(query)
    query_vector = vectorizer.transform([query_tokens])
    similarity = SimilarityEngine()
    scores = similarity.query_similarity(query_vector, sentence_matrix)

    ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
    return ranked[:5]


def collect_latest_phase2_contexts():
    downloads_dir = Path("downloaded_files")
    if not downloads_dir.exists():
        raise AssertionError("downloaded_files/ directory is missing")

    html_files = sorted(downloads_dir.glob("*.html"))
    if not html_files:
        raise AssertionError("No HTML files found in downloaded_files/. Run Phase 1 first.")

    batches: Dict[str, List[Path]] = {}
    for file in html_files:
        ts = _parse_timestamp_prefix(file.name)
        if ts:
            batches.setdefault(ts, []).append(file)

    if not batches:
        raise AssertionError("No timestamped HTML files found in downloaded_files/")

    latest_prefix = sorted(batches.keys())[-1]
    latest_files = batches[latest_prefix]
    older_files = [file for ts, files in batches.items() if ts != latest_prefix for file in files]

    assert len(latest_files) == 5, f"Expected 5 files in latest batch {latest_prefix}, found {len(latest_files)}"

    for file in older_files:
        try:
            file.unlink()
        except Exception as exc:
            raise AssertionError(f"Failed to delete older file {file}: {exc}") from exc

    remaining = sorted(downloads_dir.glob("*.html"))
    if set(remaining) != set(latest_files):
        raise AssertionError("Cleanup failed: older files still present in downloaded_files/")

    html_extractor = HTMLExtractor()
    pipeline = ProcessorPipeline(
        validators=None,
        extractor=html_extractor.extract,
        summarizer=None,
        question_generator=None,
    )

    processor = TextProcessor()
    contexts = []

    for file in latest_files:
        document = DocumentRecord(
            id=file.stem,
            title=None,
            media_type="text/html",
            path=file,
            source_path=file,
        )
        context = pipeline.run(document)
        if context.errors:
            raise AssertionError(f"Pipeline errors for {file.name}: {context.errors}")
        if context.extraction is None:
            raise AssertionError(f"Extraction missing for {file.name}")

        raw_text = context.extraction.raw_text
        clean_text = context.extraction.clean_text
        if not isinstance(raw_text, str) or len(raw_text) <= 200:
            raise AssertionError(f"Raw text too short for {file.name}")
        if not isinstance(clean_text, str) or len(clean_text) <= 200:
            raise AssertionError(f"Clean text too short for {file.name}")

        lowered = clean_text.lower()
        if "machine" not in lowered:
            raise AssertionError(f"'machine' not found in clean text for {file.name}")
        if "learn" not in lowered:
            raise AssertionError(f"'learn' not found in clean text for {file.name}")

        processed_tokens = processor.process(clean_text)
        if len(processed_tokens) < 50:
            raise AssertionError(f"Too few processed tokens for {file.name} ({len(processed_tokens)} tokens)")

        ranked_sentences = _rank_sentences_with_fallback(clean_text, "machine learning", processor)
        if not ranked_sentences:
            raise AssertionError(f"No ranked sentences produced for {file.name}")
        top_sentence, _ = ranked_sentences[0]
        lowered_sentence = top_sentence.lower()
        if "machine" not in lowered_sentence or "learn" not in lowered_sentence:
            raise AssertionError(f"Top ranked sentence for {file.name} is off-topic: {top_sentence!r}")

        contexts.append(
            {
                "document": document,
                "context": context,
                "processed_tokens": processed_tokens,
                "ranked_sentences": ranked_sentences,
            }
        )

    vectorizer = TextVectorizer()
    document_matrix = vectorizer.fit_transform([c["processed_tokens"] for c in contexts])

    top_keywords_per_doc = vectorizer.extract_top_keywords(document_matrix, top_k=20)
    for entry, keywords in zip(contexts, top_keywords_per_doc):
        file = entry["document"].path
        tag = _domain_tag(file.name)
        expected_terms = _expected_token_hints(tag)
        top_terms = {term for term, _ in keywords}
        if not any(term in top_terms for term in expected_terms):
            raise AssertionError(
                f"No expected keywords found for {tag} in {file.name}. "
                f"Top terms: {sorted(top_terms)[:10]}"
            )

    similarity_engine = SimilarityEngine()
    query_tokens = processor.process("machine learning")
    query_vector = vectorizer.transform([query_tokens])
    relevance_scores = similarity_engine.query_similarity(query_vector, document_matrix)
    if len(relevance_scores) != 5:
        raise AssertionError("Expected 5 similarity scores")

    for entry, score in zip(contexts, relevance_scores):
        file = entry["document"].path
        if score < 0.05:
            raise AssertionError(f"{file.name} has low relevance score ({score:.4f})")

    ranked_indices = sorted(range(len(relevance_scores)), key=lambda i: relevance_scores[i], reverse=True)
    top_three = [contexts[i]["document"].path.name.lower() for i in ranked_indices[:3]]
    if not (any("ibm" in name for name in top_three) or any("wikipedia" in name for name in top_three)):
        raise AssertionError(f"Expected IBM or Wikipedia among top-3 documents, got: {top_three}")

    return contexts, processor


@pytest.mark.integration
def test_phase2_processing_pipeline_end_to_end():
    contexts, _ = collect_latest_phase2_contexts()
    assert len(contexts) == 5, "Phase 2 should return exactly five document contexts"
