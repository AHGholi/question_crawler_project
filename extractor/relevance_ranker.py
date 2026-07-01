"""Backward-compatible wrapper for the moved relevance ranker module.

The implementation now lives in the processor package, but this shim keeps
existing imports from extractor.relevance_ranker working.
"""

from __future__ import annotations

from processor.relevance_ranker import RelevanceRanker

__all__ = ["RelevanceRanker"]
