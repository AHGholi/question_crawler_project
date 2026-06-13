from __future__ import annotations

import argparse
import logging

from main import run_main_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the exam crawler question generation pipeline")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--max-results", type=int, default=5, help="Maximum search results to crawl")
    parser.add_argument("--limit", type=int, default=0, help="Document processing limit (0 = no limit)")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    limit = None if args.limit == 0 else args.limit
    result = run_main_pipeline(query=args.query, max_results=args.max_results, limit=limit)

    print("Processed documents:", result.stats.processed_documents)
    print("Failed documents:", result.stats.failed_documents)
    print("Total questions:", result.stats.total_questions)
    if result.stats.failed_documents:
        print("\nFailures:")
        for ctx in result.contexts:
            if ctx.errors:
                print(f"- {ctx.document.id}: {ctx.errors}")


if __name__ == "__main__":
    main()
