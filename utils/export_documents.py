# utils/export_documents.py
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml  # requires PyYAML

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
if CONFIG_PATH.is_file():
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        try:
            config = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise RuntimeError(f"Could not parse {CONFIG_PATH}: {exc}") from exc

    google_cfg = config.get("google", {})
    api_key = google_cfg.get("api_key")
    cse_id = google_cfg.get("cse_id") or google_cfg.get("cse_cx")

    if api_key:
        os.environ.setdefault("GOOGLE_API_KEY", api_key)
    if cse_id:
        os.environ.setdefault("GOOGLE_CSE_ID", cse_id)
else:
    print(f"Warning: {CONFIG_PATH} not found; expecting credentials in environment.")

from main import crawl_documents
from utils.models import DocumentRecord




def document_to_dict(document: DocumentRecord) -> Dict[str, Any]:
    """
    Convert a DocumentRecord instance into a JSON-serializable dict.
    Supports pydantic, dataclasses, __slots__, and plain objects.
    """
    # Pydantic v2
    if hasattr(document, "model_dump"):
        data = document.model_dump()  # type: ignore[attr-defined]

    # Pydantic v1
    elif hasattr(document, "dict"):
        data = document.dict()  # type: ignore[attr-defined]

    # Dataclasses
    elif hasattr(document, "__dataclass_fields__"):
        from dataclasses import asdict
        data = asdict(document)

    # Slots-only classes
    elif hasattr(document, "__slots__"):
        data = {slot: getattr(document, slot) for slot in document.__slots__}

    # Plain objects
    elif hasattr(document, "__dict__"):
        data = vars(document)

    else:
        # Last resort: try to access known fields
        try:
            data = {
                "id": document.id,
                "title": document.title,
                "path": document.path,
                "metadata": document.metadata,
            }
        except Exception as exc:
            raise TypeError(
                "DocumentRecord type is not serializable by default."
            ) from exc

    # Normalize path-like fields
    if "path" in data and not isinstance(data["path"], str):
        data["path"] = str(data["path"])
    if "source_path" in data and not isinstance(data["source_path"], str):
        data["source_path"] = str(data["source_path"])

    return data



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export DocumentRecord payloads from the crawler for integration tests."
    )
    parser.add_argument("query", help="Search query to pass to the crawler.")
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum number of search results to download (default: 5).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tests/fixtures/phase1/sample_documents.json"),
        help="Output path for the DocumentRecord JSON fixture.",
    )
    args = parser.parse_args()

    records = [
        document_to_dict(doc)
        for doc in crawl_documents(args.query, max_results=args.max_results)
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(records)} DocumentRecord entries to {args.out}")


if __name__ == "__main__":
    main()
