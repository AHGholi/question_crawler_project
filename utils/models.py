# utils/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass(slots=True)
class DocumentRecord:
    id: str
    title: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    media_type: str | None = None
    path: Path | None = None
    source_path: Path | None = None 
    content: str | None = None
    encoding: str | None = None

    def __post_init__(self) -> None:
        # Normalize path types
        if self.path is not None and not isinstance(self.path, Path):
            self.path = Path(self.path)

        if self.source_path is not None and not isinstance(self.source_path, Path):
            self.source_path = Path(self.source_path)

        # Keep path/source_path in sync
        if self.path is None and self.source_path is not None:
            self.path = self.source_path
        elif self.source_path is None and self.path is not None:
            self.source_path = self.path
            
    @property
    def url(self) -> str | None:
        if self.source_path:
            return str(self.source_path)
        if self.path:
            return str(self.path)
        return None

    @property
    def timestamp(self) -> Optional[str]:
         return self.metadata.get("timestamp")
    


@dataclass(slots=True)
class ExtractionResult:
    document: DocumentRecord
    raw_text: str
    clean_text: str
    tokens: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    keyword_scores: Dict[str, float] = field(default_factory=dict)
    summary: Optional[str] = None
    errors: List[str] = field(default_factory=list)

    # NEW optional fields for ranking / question generation
    top_sentences: List[str] = field(default_factory=list)
    sentence_scores: Dict[str, float] = field(default_factory=dict)

    questions: List["QuestionItem"] = field(default_factory=list)


    @property
    def document_id(self) -> str:
        return self.document.id


@dataclass(slots=True)
class Question:
    prompt: str
    answer: Optional[str] = None
    difficulty: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class QuestionSet:
    document: DocumentRecord
    questions: List["QuestionItem"]
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    strategy: Optional[str] = None

@dataclass
class QuestionItem:
    """
    Canonical shape for pipeline-generated question/answer pairs.
    """
    question: str
    answer: str
    source_document_id: str
    source_sentence: str
    confidence: float
    tags: List[str] = field(default_factory=list)


__all__ = ["DocumentRecord", "ExtractionResult", "Question", "QuestionSet", "QuestionItem"]
