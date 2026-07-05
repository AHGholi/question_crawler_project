# ui\cli.py

"""Command-line interface for running the exam-crawler workflow.

This module exposes a small terminal-based entry point for invoking the same
pipeline that the Streamlit UI uses.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Iterable

from main import run_main_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pipeline runner."""
    parser = argparse.ArgumentParser(description="Run the exam crawler question generation pipeline")
    parser.add_argument("--query", help="Search query", default=None)
    parser.add_argument("--input-files", nargs="+", help="Local text or HTML files to process")
    parser.add_argument("--max-results", type=int, default=5, help="Maximum search results to crawl")
    parser.add_argument("--limit", type=int, default=0, help="Document processing limit (0 = no limit)")
    parser.add_argument("--show-qa", action="store_true", help="Print generated question/answer pairs")

    parser.add_argument(
        "--qg-provider",
        choices=["local_hf", "hf_api"],
        default="local_hf",
        help="Question-generation backend",
    )
    parser.add_argument(
        "--qg-model",
        default="google/flan-t5-large",
        help="Hugging Face model name",
    )
    parser.add_argument(
        "--hf-api-token",
        default=None,
        help="Hugging Face API token (optional; env var can also be used)",
    )

    return parser.parse_args()


def _question_text(q: Any) -> str:
    """Return a question string from a question-like object."""
    return (getattr(q, "question", None) or getattr(q, "prompt", None) or "").strip()


def _question_answer(q: Any) -> str:
    """Return an answer string from a question-like object."""
    return (getattr(q, "answer", None) or "").strip()


def _iter_questions(ctx: Any) -> Iterable[Any]:
    """Yield the question objects stored in a pipeline context."""
    if not ctx or not getattr(ctx, "questions", None):
        return []
    items = getattr(ctx.questions, "questions", None)
    return items if items else []


def _safe_title(ctx: Any) -> str:
    """Return a readable title for a document context in the terminal output."""
    doc = getattr(ctx, "document", None)
    title = getattr(doc, "title", None) if doc else None
    if title and str(title).strip():
        return str(title).strip()
    return f"Document {getattr(doc, 'id', 'unknown')}"


def main() -> None:
    """Execute the CLI workflow and print the generated question summary."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    if not args.query and not args.input_files:
        raise SystemExit("Either --query or --input-files must be provided.")
    if args.query and args.input_files:
        raise SystemExit("Provide only one of --query or --input-files.")

    limit = None if args.limit == 0 else args.limit
    result = run_main_pipeline(
        query=args.query,
        max_results=args.max_results,
        limit=limit,
        input_files=args.input_files,
        qg_provider=args.qg_provider,
        qg_model=args.qg_model,
        qg_api_token=args.hf_api_token,
    )


    print("Processed documents:", result.stats.processed_documents)
    print("Failed documents:", result.stats.failed_documents)
    print("Total questions:", result.stats.total_questions)

    if args.show_qa:
        print("\nGenerated Q/A:")
        any_printed = False
        for i, ctx in enumerate(result.contexts, start=1):
            qs = list(_iter_questions(ctx))
            if not qs:
                continue
            any_printed = True
            print(f"\n{i}. {_safe_title(ctx)}")
            for j, q in enumerate(qs, start=1):
                q_text = _question_text(q)
                q_ans = _question_answer(q)
                print(f"  Q{j}: {q_text}")
                print(f"  A{j}: {q_ans if q_ans else '[No answer]'}")

        if not any_printed:
            print("No questions generated.")

    if result.stats.failed_documents:
        print("\nFailures:")
        for ctx in result.contexts:
            if ctx.errors:
                print(f"- {ctx.document.id}: {ctx.errors}")


if __name__ == "__main__":
    main()
