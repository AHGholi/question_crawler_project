"""Core abstractions for the question-generation subsystem.

The classes in this module define the input payload passed to question
backends and the protocol that concrete generators implement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol


@dataclass(slots=True)
class QGInput:
    """Structured input passed to a question generator backend."""

    topic: str
    title: str | None = None
    summary: str | None = None
    keywords: List[str] = field(default_factory=list)
    sentences: List[str] = field(default_factory=list)
    chunks: List[str] = field(default_factory=list)
    num_questions: int = 10


class QGBackend(Protocol):
    """Protocol for question-generation backends."""

    def generate(self, data: QGInput) -> List[str]:
        """Return a list of generated question strings for the supplied input."""
        ...
