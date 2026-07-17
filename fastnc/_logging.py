"""Internal helpers for structured FASTNC logging.

FASTNC never configures handlers or logging levels.  Applications control the
``fastnc`` logger hierarchy with the standard :mod:`logging` API.
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
import logging
import time
from typing import Callable, ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")
_depth: ContextVar[int] = ContextVar("fastnc_log_depth", default=0)


def log(logger: logging.Logger, level: int, message: str, *args) -> None:
    """Emit one indented log record in the current FASTNC call context."""
    if logger.isEnabledFor(level):
        logger.log(level, "%s" + message, "  " * _depth.get(), *args)


def log_call(
    logger: logging.Logger,
    level: int = logging.INFO,
    *,
    timing_key: str | None = None,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Log entry, exit, elapsed time, and exceptions for a function call."""
    def decorate(func: Callable[_P, _R]) -> Callable[_P, _R]:
        name = func.__qualname__

        @wraps(func)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            enabled = logger.isEnabledFor(level)
            depth = _depth.get()
            if enabled:
                logger.log(level, "%s[%s] started", "  " * depth, name)
            token = _depth.set(depth + 1)
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            except Exception:
                logger.exception("%s[%s] failed", "  " * depth, name)
                raise
            finally:
                elapsed = time.perf_counter() - start
                if timing_key is not None and args:
                    timings = getattr(args[0], "timings", None)
                    if isinstance(timings, dict):
                        timings[timing_key] = elapsed
                _depth.reset(token)
                if enabled:
                    logger.log(
                        level,
                        "%s[%s] finished in %.3f s",
                        "  " * depth,
                        name,
                        elapsed,
                    )

        return wrapped

    return decorate
