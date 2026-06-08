# crawler/downloader.py

from __future__ import annotations
import os
import logging
import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 15


def download_file(url: str, output_path: str) -> str:
    """
    Download URL to output_path and return the output_path.
    Raises exceptions on HTTP errors.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    headers = {"User-Agent": os.getenv("USER_AGENT", "exam_crawler/1.0")}
    response = requests.get(url, headers=headers, stream=True, timeout=DEFAULT_TIMEOUT)

    try:
        response.raise_for_status()
        ctype = response.headers.get("Content-Type", "")
        if not ("text/html" in ctype or "application/xhtml+xml" in ctype or "text/plain" in ctype):
            raise ValueError(f"Refusing to download non-HTML content-type: {ctype}")

        with open(output_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
    finally:
        response.close()

    logger.info("Saved %s -> %s", url, output_path)
    return output_path
