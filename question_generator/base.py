from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol


@dataclass(slots=True)
class QGInput:
    topic: str
    title: str | None = None
    summary: str | None = None
    keywords: List[str] = field(default_factory=list)
    sentences: List[str] = field(default_factory=list)
    chunks: List[str] = field(default_factory=list)  # NEW
    num_questions: int = 10


class QGBackend(Protocol):
    def generate(self, data: QGInput) -> List[str]:
        """Return list of question strings."""
        ...
