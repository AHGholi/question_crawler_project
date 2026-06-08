# utils/retry.py
from __future__ import annotations
import time
from typing import Callable, Tuple, Type, TypeVar

T = TypeVar("T")
ExceptionTypes = Tuple[Type[BaseException], ...]

def call_with_retry(
    operation: Callable[[], T],
    *,
    tries: int = 3,
    delay: float = 0.5,
    backoff: float = 2.0,
    exceptions: ExceptionTypes = (Exception,)
) -> T:
    attempt = 0
    wait = delay
    while True:
        attempt += 1
        try:
            return operation()
        except exceptions:
            if attempt >= tries:
                raise
            time.sleep(wait)
            wait *= backoff
