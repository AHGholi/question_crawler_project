# utils/errors.py
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator, Type


class CrawlerError(Exception):
    """Base class for all domain-specific failures."""
    exit_code = 1


class ConfigurationError(ValueError, CrawlerError):
    exit_code = 2


class SearchError(CrawlerError):
    exit_code = 3


class DownloadError(CrawlerError):
    exit_code = 4


class ExtractionError(CrawlerError):
    exit_code = 5


class ProcessingError(CrawlerError):
    exit_code = 6


@contextmanager
def rethrow_as(error_cls: Type[CrawlerError], message: str) -> Iterator[None]:
    """
    Wrap a block to convert any exception into the supplied domain error.
    Existing instances of that class are re-raised unchanged.
    """
    try:
        yield
    except Exception as exc:  # pylint: disable=broad-except
        if isinstance(exc, error_cls):
            raise
        raise error_cls(message) from exc


def handle_exception(
    logger: logging.Logger,
    exc: Exception,
    *,
    exit_on_error: bool = False,
    default_exit_code: int = 1,
) -> None:
    """
    Central logging + exit-code handler for both CLI and library callers.
    """
