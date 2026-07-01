# Question Crawler Project

This repository implements a document crawl → extract → process → question generation pipeline.

## Current runtime flow

The active question-generation path is:

1. [main.py](main.py) chooses either a web search crawler or a local-file crawler.
2. [main_pipeline.py](main_pipeline.py) runs each discovered document through the processing pipeline.
3. The extractor layer produces an [utils/models.py](utils/models.py) extraction result from HTML or text input.
4. The processor layer ranks sentences, extracts keywords, and prepares context for question generation.
5. The question generation adapter in [question_generator/adapter.py](question_generator/adapter.py) builds the input payload and filters the generated questions.
6. The actual backend is either:
   - [question_generator/local_hf_qg.py](question_generator/local_hf_qg.py) for a locally loaded Hugging Face model, or
   - [question_generator/hf_qg.py](question_generator/hf_qg.py) for the Hugging Face inference API.

## Project structure

- [crawler](crawler): search and download logic, URL filtering, robots handling, and API client adapters.
- [extractor](extractor): HTML, text, PDF, and document extraction plus cleaning helpers.
- [processor](processor): sentence ranking, tokenization, vectorization, similarity, and relevance scoring.
- [question_generator](question_generator): the current question-generation adapter and backends.
- [ui](ui): Streamlit app and CLI wrapper.
- [utils](utils): shared models, configuration helpers, retry logic, and text utilities.
- [downloaded_files](downloaded_files): downloaded HTML pages produced during crawler runs.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the pipeline

```powershell
python main.py --query "machine learning"
```

## Process local files

```powershell
python main.py --input-files sample.txt
python ui/cli.py --input-files sample.txt
```

## Run the CLI wrapper

```powershell
python ui/cli.py --query "machine learning"
```

## Run the Streamlit UI

```powershell
streamlit run ui/app.py
```

## Notes on unused or non-runtime helpers

- [utils/export_documents.py](utils/export_documents.py) is an auxiliary export helper for creating JSON fixtures. It is not part of the standard runtime question-generation path and is only relevant if you need fixture export for testing or integration work.
- The repository currently uses the adapter-based question-generation flow in [question_generator/adapter.py](question_generator/adapter.py). Older or stale references to a legacy generator module were found only in tests and do not correspond to a runtime module in the current codebase.
