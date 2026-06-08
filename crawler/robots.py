# crawler/robots.py
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 3600  # refresh robots.txt once per hour
DEFAULT_TIMEOUT = 10

@dataclass
class RobotsEntry:
    parser: Optional[RobotFileParser]
    fetched_at: float
    allow_all: bool

_CACHE: Dict[str, RobotsEntry] = {}

def _robots_url_for(url: str) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme or "http"
    if not parsed.netloc:
        raise ValueError(f"Cannot derive robots URL for: {url}")
    return urlunparse((scheme, parsed.netloc, "/robots.txt", "", "", ""))

def _load_entry(url: str, user_agent: str) -> RobotsEntry:
    robots_url = _robots_url_for(url)
    cached = _CACHE.get(robots_url)
    now = time.time()
    if cached and now - cached.fetched_at < CACHE_TTL_SECONDS:
        return cached

    headers = {"User-Agent": user_agent}
    try:
        response = requests.get(robots_url, headers=headers, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        logger.info("robots.txt fetch failed for %s: %s", robots_url, exc)
        entry = RobotsEntry(parser=None, fetched_at=now, allow_all=True)
    else:
        if response.status_code >= 400 or not response.text.strip():
            logger.debug("robots.txt missing at %s (status %s)", robots_url, response.status_code)
            entry = RobotsEntry(parser=None, fetched_at=now, allow_all=True)
        else:
            parser = RobotFileParser()
            parser.set_url(robots_url)
            parser.parse(response.text.splitlines())
            entry = RobotsEntry(parser=parser, fetched_at=now, allow_all=False)

    _CACHE[robots_url] = entry
    return entry

def is_allowed(url: str, user_agent: str) -> bool:
    try:
        entry = _load_entry(url, user_agent)
    except ValueError:
        return False  # malformed URL
    if entry.allow_all or entry.parser is None:
        return True
    allowed = entry.parser.can_fetch(user_agent, url)
    if not allowed:
        logger.info("robots.txt disallows %s for %s", url, user_agent)
    return allowed

def _to_optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            logger.warning("Invalid crawl-delay value %r", value)
            return None
    return None

def get_crawl_delay(url: str, user_agent: str) -> Optional[float]:
    try:
        entry = _load_entry(url, user_agent)
    except ValueError:
        return None

    if entry.parser is None:
        return None

    delay_any = entry.parser.crawl_delay(user_agent)
    if delay_any is None:
        delay_any = entry.parser.crawl_delay("*")

    return _to_optional_float(delay_any)