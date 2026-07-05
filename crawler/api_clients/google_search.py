# crawler\api_clients\google_search.py

"""Google Custom Search client used by the crawler.

This module retrieves search results from Google's Custom Search API so the
crawler can discover candidate web pages to download.
"""

from __future__ import annotations
import os
from typing import Dict, List, Tuple
import requests
from dotenv import load_dotenv

load_dotenv()

CSE_URL = "https://www.googleapis.com/customsearch/v1"


def _get_credentials() -> Tuple[str, str]:
    """Load and validate the API credentials required for the search request."""
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID") or os.getenv("GOOGLE_CSE_CX")
    if not api_key or not cse_id:
        raise ValueError(
            "Missing Google credentials. Set environment variables GOOGLE_API_KEY and GOOGLE_CSE_ID (or GOOGLE_CSE_CX).\n"
            "Example .env:\n"
            "  GOOGLE_API_KEY=...\n"
            "  GOOGLE_CSE_ID=...\n"
        )
    return api_key, cse_id


def _google_search_page(
    query: str,
    num_results: int,
    start: int,
    api_key: str,
    cse_id: str
) -> List[Dict]:
    """Fetch one page of search results from the Google Custom Search API."""
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": max(1, min(10, num_results)),
        "start": max(1, int(start))
    }
    headers = {"User-Agent": os.getenv("USER_AGENT", "exam_crawler/1.0")}

    response = requests.get(CSE_URL, params=params, headers=headers, timeout=15)
    response.raise_for_status()

    data = response.json()
    items = data.get("items", []) or []
    return [{"title": it.get("title"), "link": it.get("link"), "snippet": it.get("snippet")} for it in items]


def google_search(query: str, num_results: int = 10, start: int = 1) -> List[Dict]:
    """Retrieve search results while handling pagination across multiple API calls."""
    api_key, cse_id = _get_credentials()

    remaining = max(1, int(num_results))
    next_start = max(1, int(start))

    results: List[Dict] = []
    while remaining > 0:
        batch_size = min(10, remaining)
        page_items = _google_search_page(
            query,
            num_results=batch_size,
            start=next_start,
            api_key=api_key,
            cse_id=cse_id
        )
        results.extend(page_items)

        if len(page_items) < batch_size:
            break

        remaining -= len(page_items)
        next_start += len(page_items)

    return results[:num_results]
