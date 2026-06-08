# utils/text.py
from __future__ import annotations
import re
from datetime import datetime, UTC
from typing import Optional

_SLUG_RE = re.compile(r"[^a-z0-9]+")

def slugify(value: Optional[str], fallback: str = "document") -> str:
    text = (value or "").lower()
    text = _SLUG_RE.sub("-", text).strip("-")
    return text or fallback

def timestamp_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
