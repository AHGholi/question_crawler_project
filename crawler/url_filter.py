# crawler/url_filter.py
from __future__ import annotations
from typing import Optional, List
from urllib.parse import urlparse, urlunparse

BLACKLISTED_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp",
    ".mp4", ".mp3", ".avi", ".mov", ".wmv", ".zip", ".rar",
    ".exe", ".iso"
)

ALLOWED_SCHEMES = ("http", "https")

def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = (parsed.scheme or "http").lower()
    netloc = parsed.netloc.lower()

    # Remove default ports (80 for http, 443 for https)
    host, sep, port = netloc.partition(":")
    if (scheme == "http" and port == "80") or (scheme == "https" and port == "443"):
        netloc = host

    path = parsed.path or "/"
    query = parsed.query

    return urlunparse((scheme, netloc, path, "", query, ""))  # drop fragment

def is_valid_scheme(url: str) -> bool:
    p = urlparse(url)
    return p.scheme in ALLOWED_SCHEMES

def has_disallowed_extension(url: str) -> bool:
    u = url.lower()
    return any(u.endswith(ext) for ext in BLACKLISTED_EXTENSIONS)

def candidate_allowed(
    url: str,
    allow_domains: Optional[List[str]] = None,
    deny_domains: Optional[List[str]] = None
) -> bool:
    """
    True if URL passes basic checks.
    - allow_domains: if provided, only netlocs that endwith one entry are allowed.
    - deny_domains: if provided, any netloc containing an entry is rejected.
    """
    if not url:
        return False
    if not is_valid_scheme(url):
        return False
    if has_disallowed_extension(url):
        return False

    p = urlparse(url)
    netloc = (p.netloc or "").lower()

    if deny_domains:
        for d in deny_domains:
            if d.lower() in netloc:
                return False

    if allow_domains:
        for d in allow_domains:
            if netloc.endswith(d.lower()):
                return True
        return False

    return True
