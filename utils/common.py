# utils/common.py
from __future__ import annotations

import re
from typing import List


def simple_sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text or "")
    return [p.strip() for p in parts if len(p.strip()) > 5]
