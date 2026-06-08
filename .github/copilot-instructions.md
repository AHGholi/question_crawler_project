# Exam Crawler Project - AI Assistant Instructions

## Architecture Overview

This is a **document crawl → extract → process → question generation pipeline**. Three-phase design:

1. **Crawl** (`crawler/`): Search queries via Google/Bing APIs, download HTML/PDF/DOCX files to `downloaded_files/`.
2. **Extract & Process** (`extractor/` + `processor/`): Parse content, rank sentences, vectorize text.
3. **Question Generation** (`question_generator/`): Generate exam-style Q&A pairs (neural + rule-based engines).

Orchestration: `MainPipeline` ([main_pipeline.py](main_pipeline.py)) iterates `CrawlStep` output, feeds each `DocumentRecord` through `ProcessorPipeline`, collects results + stats.

## Critical Data Models

- **`DocumentRecord`** ([utils/models.py](utils/models.py#L1-L30)): Represents a downloadable doc (path, title, metadata, encoding). Created by crawler adapter in [main.py](main.py#L1-L40).
- **`ExtractionResult`** ([utils/models.py](utils/models.py#L1-L60)): Enriched document (raw_text, clean_text, keywords, top_sentences, questions).
- **`PipelineContext`**: Flows through processor pipeline; holds errors and final `QuestionSet`.
- **`QuestionItem`** ([utils/models.py](utils/models.py#L60-L75)): Canonical Q&A shape (question, answer, confidence, source_document_id, source_sentence).

## Key Files by Responsibility

| Component | Core Files | Purpose |
|-----------|-----------|---------|
| **Crawl** | `crawler/search.py`, `crawler/downloader.py`, `crawler/api_clients/*` | Query APIs, manage downloads, enforce robots.txt |
| **Extract** | `extractor/html_extractor.py`, `extractor/pdf_extractor.py`, `extractor/cleaner.py` | Parse formats, clean text |
| **Rank & Process** | `processor/pipeline.py`, `processor/sentence_ranking.py`, `processor/vectorizer.py` | Score sentences, vectorize, compute similarity |
| **Question Gen** | `question_generator/generator.py`, `question_generator/neural_qg.py`, `rule_based_QEngine/*` | **[DEPRECATED]** To be replaced; see note below. |
| **Config & Logging** | `utils/config.py`, `utils/logger.py`, `utils/models.py` | YAML config loading, centralized logging, data models |

## Configuration & Workflows

- **Config**: YAML file at `config.yaml` loaded via `utils/config.py`. Env vars override via `.env` (using `python-dotenv`).
  - Key settings: crawler delays, Google API credentials, question generation strategy (neural/rule-based), model names.
- **Logging**: Use `utils/logger.py` for consistent logs (required in new modules).
- **Tests**: Run `pytest tests/` (markers: `@pytest.mark.integration`, `@pytest.mark.network`). Fixtures in `tests/fixtures/phase1/`.
  - Integration tests ([tests/integration/](tests/integration/)) are canonical examples of expected behavior.

## Development Patterns

1. **Pipeline Protocol**: Small callable steps composed together. `CrawlStep` protocol ([main_pipeline.py](main_pipeline.py#L1-L20)) yields `DocumentRecord`s; extractors and rankers are functions/callables passed to `ProcessorPipeline`.

2. **Error Tracking**: `PipelineContext.errors` is a list; `ProcessingStats.observe_context()` detects and counts failures. Always append to context.errors, never raise uncaught exceptions from pipeline steps.

3. **Dual Question Engines**: Currently coexist (neural + rule-based). Rule-based engines ([question_generator/rule_based_QEngine/](question_generator/rule_based_QEngine/)) define question types: `definition.py`, `factoid.py`, `listing.py`.

4. **Extraction Hierarchy**: All extractors inherit from `extractor/base.py` (`BaseExtractor` interface with `supports()` and `extract()` methods). Implement new formats there.

## ⚠️ Important: Question Generator Deprecation

The `question_generator/` module **will be deleted and replaced with an external library**. 

**Impact on new work**:
- Do NOT expand the neural_qg or rule_based_QEngine logic.
- If changes are needed, keep them minimal until external lib is integrated.
- When migrating: Ensure the external lib outputs compatible `QuestionItem` objects ([utils/models.py](utils/models.py#L1-L75)) so `PipelineContext` and downstream code remain unchanged.

## Quick Dev Commands

```bash
# Setup
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt

# Run tests
pytest tests/ -v
pytest tests/ -m integration  # integration tests only

# Run pipeline
python main.py

# With CLI (if implemented)
python ui/cli.py --help
```

## Integration Points & External Dependencies

- **Search APIs**: Google Custom Search (API key in `config.yaml`); fallback to Bing in `api_clients/`.
- **ML Models**: `transformers` library for neural QG (model name in `config.yaml`).
- **NLP**: `spacy` for tokenization/NER; `torch` for embeddings.
- **Rendering**: `reportlab` for PDF parsing.
- **UI**: `streamlit` for web interface (`ui/web_ui.py`).

## File Organization by Phase

- **Phase 1 (Crawl)**: `crawler/`, outputs to `downloaded_files/`
- **Phase 2 (Extract+Process)**: `extractor/`, `processor/` read from `downloaded_files/`, enrich `ExtractionResult`
- **Phase 3 (Question Gen)**: `question_generator/` produces `QuestionItem`s → `QuestionSet`
- **Shared**: `utils/`, `config.yaml`, `.env`
- **UI/Output**: `ui/` directory

---

**Project not yet committed to repository.** When setting up version control, include `.github/workflows/` for CI (test on push) and `.gitignore` (exclude `.venv/`, `downloaded_files/`, `.env`).
