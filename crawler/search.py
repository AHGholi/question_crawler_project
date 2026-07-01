"""Search and download orchestration for the crawler subsystem.

This module collects results from the configured search backend, filters them
according to domain and file-type rules, and downloads accepted pages into the
project's download directory.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from utils.config import get_setting
from utils.text import slugify, timestamp_slug

logger = logging.getLogger(__name__)

ALLOW_DOMAINS: Optional[List[str]] = get_setting("crawler", "allowed_domains")
RAW_DENY_EXTENSIONS: Optional[List[str]] = get_setting("crawler", "denied_extensions")
CRAWL_DELAY_SECONDS: float = float(get_setting("crawler", "crawl_delay_seconds", default=0.35))
DOWNLOAD_DIR = get_setting("crawler", "download_dir", default="downloaded_files")
MAX_WORKERS: int = max(1, int(get_setting("crawler", "max_workers", default=4)))
USER_AGENT = get_setting("crawler", "user_agent", default=os.getenv("USER_AGENT", "exam_crawler/1.0"))

DEFAULT_DENY_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".svg",
    ".webp",
    ".mp4",
    ".mp3",
    ".avi",
    ".mov",
    ".wmv",
    ".zip",
    ".rar",
    ".exe",
    ".iso",
]


def _normalize_extensions(configured: Optional[List[str]]) -> List[str]:
    """Normalize configured file extensions to a consistent lowercase form."""
    source = configured or DEFAULT_DENY_EXTENSIONS
    normalized: List[str] = []
    for ext in source:
        if not ext:
            continue
        e = ext.lower()
        if not e.startswith("."):
            e = f".{e}"
        normalized.append(e)
    return normalized


DENY_EXTENSIONS: List[str] = _normalize_extensions(RAW_DENY_EXTENSIONS)

try:
    from crawler.api_clients.google_search import google_search
    from crawler.url_filter import normalize_url, candidate_allowed
    from crawler.downloader import download_file
except Exception:
    from api_clients.google_search import google_search
    from url_filter import normalize_url, candidate_allowed
    from downloader import download_file

try:
    from crawler import robots  # type: ignore
except Exception:
    try:
        import robots  # type: ignore
    except Exception:
        robots = None  # type: ignore

_domain_lock = threading.Lock()
_domain_next_allowed: Dict[str, float] = {}


def _has_denied_extension(url: str) -> bool:
    """Return True when the URL points to a file type that should be skipped."""
    if not DENY_EXTENSIONS:
        return False
    lowered = url.lower()
    return any(lowered.endswith(ext) for ext in DENY_EXTENSIONS)


def _effective_delay(url: str) -> float:
    """Resolve the crawl delay for a URL, preferring robots rules when available."""
    delay = None
    if robots and hasattr(robots, "get_crawl_delay"):
        try:
            delay = robots.get_crawl_delay(url, user_agent=USER_AGENT)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.debug("robots delay lookup failed for %s: %s", url, exc)
    value = delay if delay is not None else CRAWL_DELAY_SECONDS
    return max(0.0, float(value))


def _reserve_slot(url: str) -> None:
    """Ensure a domain is not crawled too aggressively by enforcing a delay."""
    netloc = urlparse(url).netloc.lower() or "default"
    delay = _effective_delay(url)
    while True:
        with _domain_lock:
            now = time.monotonic()
            ready_at = _domain_next_allowed.get(netloc, now)
            sleep_for = ready_at - now
            if sleep_for <= 0:
                _domain_next_allowed[netloc] = now + delay
                return
        time.sleep(max(sleep_for, 0.0))


def _download_candidate(task: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """Download one candidate URL and package the result into a search payload."""
    url = task["url"]
    _reserve_slot(url)
    try:
        saved = download_file(url, task["output_path"])
        dl_meta = {"url": url, "path": saved, "status": "ok"}
        logger.info("Downloaded %s", saved)
    except Exception as exc:
        dl_meta = {"url": url, "path": None, "status": "error", "reason": str(exc)}
        logger.error("Download failed for %s: %s", url, exc)

    payload = {
        "title": task["title"],
        "snippet": task["snippet"],
        "search_link": task["search_link"],
        "download": dl_meta,
    }
    return task["index"], payload


def run_search(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Collect search results, filter them, and download the accepted pages.

    The function coordinates the search backend, domain filtering, and threaded
    downloads so the pipeline receives a consistent list of metadata records.
    """
    logger.info("Searching Google for: %s", query)
    try:
        results = google_search(query, num_results=max_results)
    except Exception as exc:
        logger.error("google_search failed: %s", exc)
        raise

    if not results:
        logger.info("No results from google_search.")
        return []

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    tasks: List[Dict[str, Any]] = []
    for idx, item in enumerate(results, start=1):
        link = item.get("link")
        if not link:
            continue
        norm = normalize_url(link)

        if _has_denied_extension(norm):
            logger.debug("Skipping due to deny extension: %s", norm)
            continue

        if not candidate_allowed(norm, allow_domains=ALLOW_DOMAINS, deny_domains=None):
            logger.debug("Filtered out: %s", norm)
            continue

        title = item.get("title")
        snippet = item.get("snippet")
        title_slug = slugify(title, fallback=f"result-{idx}")
        domain_slug = slugify(urlparse(norm).netloc.split(":")[0])
        filename = f"{timestamp_slug()}_{domain_slug}_{title_slug}.html"
        out_path = os.path.join(DOWNLOAD_DIR, filename)

        tasks.append(
            {
                "index": idx,
                "title": title,
                "snippet": snippet,
                "search_link": link,
                "url": norm,
                "output_path": out_path,
            }
        )

    if not tasks:
        logger.info("No download candidates after filtering.")
        return []

    results_map: Dict[int, Dict[str, Any]] = {}
    worker_count = max(1, min(MAX_WORKERS, len(tasks)))

    if worker_count == 1:
        for task in tasks:
            idx_key, payload = _download_candidate(task)
            results_map[idx_key] = payload
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(_download_candidate, task): task for task in tasks}
            for future in as_completed(future_map):
                task = future_map[future]
                try:
                    idx_key, payload = future.result()
                except Exception as exc:
                    logger.exception("Worker failure for %s", task["url"])
                    payload = {
                        "title": task["title"],
                        "snippet": task["snippet"],
                        "search_link": task["search_link"],
                        "download": {
                            "url": task["url"],
                            "path": None,
                            "status": "error",
                            "reason": str(exc),
                        },
                    }
                    idx_key = task["index"]
                results_map[idx_key] = payload

    return [results_map[idx] for idx in sorted(results_map)]
