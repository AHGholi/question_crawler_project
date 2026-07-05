# crawler/downloader.py
"""Simple HTTP downloader used by the crawler subsystem."""

from __future__ import annotations

import logging
import os
from typing import Dict

import requests

from utils.config import get_setting

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 15
MAX_DOWNLOAD_BYTES = int(get_setting("crawler", "max_download_bytes", default=10 * 1024 * 1024))

SUPPORTED_CONTENT_TYPES = {
    "text/html": ".html",
    "application/xhtml+xml": ".html",
    "text/plain": ".txt",
    "application/pdf": ".pdf",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}


def _normalize_content_type(raw: str) -> str:
    return (raw or "").split(";", 1)[0].strip().lower()


def download_file(url: str, output_base_path: str) -> Dict[str, object]:
    """Download a URL to a local file and return structured metadata."""
    os.makedirs(os.path.dirname(output_base_path) or ".", exist_ok=True)

    headers = {"User-Agent": os.getenv("USER_AGENT", "exam_crawler/1.0")}
    response = requests.get(url, headers=headers, stream=True, timeout=DEFAULT_TIMEOUT)

    total_bytes = 0
    output_path = ""

    try:
        response.raise_for_status()

        content_type = _normalize_content_type(response.headers.get("Content-Type", ""))
        if content_type not in SUPPORTED_CONTENT_TYPES:
            raise ValueError(f"Unsupported content-type: {content_type or '<missing>'}")

        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > MAX_DOWNLOAD_BYTES:
                    raise ValueError(f"File too large: {content_length} bytes")
            except ValueError:
                pass

        extension = SUPPORTED_CONTENT_TYPES[content_type]
        output_path = output_base_path + extension

        with open(output_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                total_bytes += len(chunk)
                if total_bytes > MAX_DOWNLOAD_BYTES:
                    raise ValueError(f"Download exceeded max size ({MAX_DOWNLOAD_BYTES} bytes)")
                fh.write(chunk)

    finally:
        response.close()

    logger.info("Saved %s -> %s (%s)", url, output_path, content_type)
    return {
        "path": output_path,
        "content_type": content_type,
        "final_url": response.url,
        "bytes": total_bytes,
    }
