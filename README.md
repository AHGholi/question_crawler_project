# Exam Crawler Project

This repository is a document crawl → extract → process → question generation pipeline.

## Current runtime state

- The pipeline crawls HTML pages from search results and stores them under `downloaded_files/`.
- The extractor phase is active: `extractor/html_extractor.py` parses HTML and produces `ExtractionResult` objects.
- The processor phase is active: `processor/` ranks sentences and computes keywords.
- The question generation module is currently a placeholder; it is not using the legacy `question_generator_old/` code path.
- `question_generator_old/` is kept for reference only and is not executed by the current main pipeline.

## Project structure

- `crawler/`: search + download logic, API client adapters, URL filtering, robots handling.
- `extractor/`: HTML/PDF/text extraction and cleaning.
- `processor/`: pipeline orchestration, sentence ranking, vectorization, similarity.
- `ui/`: Streamlit app (`ui/app.py`) and CLI wrapper (`ui/cli.py`).
- `utils/`: shared data models, config loading, logging, retry helpers.
- `downloaded_files/`: downloaded HTML pages from crawler runs.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the pipeline

```powershell
python main.py "machine learning"
```

## Run the CLI wrapper

```powershell
python ui/cli.py "machine learning"
```

## Run the Streamlit UI

```powershell
streamlit run ui/app.py
```

## Notes

- `config.yaml` contains crawler settings and Google API credentials.
- The current runtime is designed to keep `question_generator_old/` out of the execution path.
- Future work should replace `question_generator/` with an external question generation library that emits `QuestionItem` objects.
