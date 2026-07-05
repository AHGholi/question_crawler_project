# utils\common.py
"""Small text-processing helpers used across the project."""
from __future__ import annotations

import re
from typing import List


def simple_sentence_split(text: str) -> List[str]:
    """Split a block of text into sentences using punctuation boundaries."""
    parts = re.split(r"(?<=[.!?])\s+", text or "")
    return [p.strip() for p in parts if len(p.strip()) > 5]
