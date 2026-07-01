#extractor/base.py
"""Base abstractions for the document extraction subsystem.

Concrete extractors implement the protocol and can be registered so the
processing pipeline can select an appropriate extractor for each document.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Protocol, Type, runtime_checkable

from utils.models import DocumentRecord, ExtractionResult


@runtime_checkable
class Extractor(Protocol):
    """Runtime protocol for extractor implementations."""

    name: str

    def supports(self, document: DocumentRecord) -> bool: ...

    def extract(self, document: DocumentRecord) -> ExtractionResult: ...


class BaseExtractor(ABC):
    """Convenience ABC that concrete extractors can inherit from."""

    #: Sub-classes must provide a stable identifier.
    name: str

    def __init_subclass__(cls, **kwargs: Any) -> None:  # type: ignore[override]
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "name", None):
            raise ValueError("Extractor subclasses must define a non-empty 'name' attribute.")

    @abstractmethod
    def supports(self, document: DocumentRecord) -> bool:
        """Return True if this extractor can handle the supplied document."""

    @abstractmethod
    def extract(self, document: DocumentRecord) -> ExtractionResult:
        """Perform the extraction and return an ExtractionResult."""


class ExtractorRegistry:
    """Registry that keeps track of available extractor implementations."""

    def __init__(self) -> None:
        self._registry: Dict[str, Type[BaseExtractor]] = {}

    def register(
        self,
        alias: str,
        extractor_cls: Type[BaseExtractor],
        *,
        replace: bool = False,
    ) -> None:
        """Register an extractor implementation under the given alias."""
        key = alias.lower()
        if key in self._registry and not replace:
            raise ValueError(f"Extractor '{alias}' is already registered.")
        self._registry[key] = extractor_cls

    def unregister(self, alias: str) -> None:
        """Remove an extractor entry – silently ignore unknown aliases."""
        self._registry.pop(alias.lower(), None)

    def get(self, alias: str) -> Type[BaseExtractor]:
        """Return the extractor class for *alias*."""
        key = alias.lower()
        try:
            return self._registry[key]
        except KeyError as exc:
            raise KeyError(f"Extractor '{alias}' is not registered.") from exc

    def create(self, alias: str, **kwargs: Any) -> BaseExtractor:
        """Instantiate and return the extractor registered under *alias*."""
        extractor_cls = self.get(alias)
        return extractor_cls(**kwargs)

    def aliases(self) -> Iterable[str]:
        """Return all registered aliases."""
        return tuple(self._registry.keys())

    def __contains__(self, alias: object) -> bool:
        return isinstance(alias, str) and alias.lower() in self._registry


registry = ExtractorRegistry()


def register_extractor(
    alias: str,
    extractor_cls: Type[BaseExtractor],
    *,
    replace: bool = False,
) -> None:
    """Convenience wrapper around the global registry."""
    registry.register(alias, extractor_cls, replace=replace)


def create_extractor(alias: str, **kwargs: Any) -> BaseExtractor:
    """Create an extractor instance from the global registry."""
    return registry.create(alias, **kwargs)


__all__ = [
    "Extractor",
    "BaseExtractor",
    "ExtractorRegistry",
    "registry",
    "register_extractor",
    "create_extractor",
]
