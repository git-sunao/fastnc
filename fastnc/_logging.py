"""Helpers for concise FASTNC progress and diagnostic logging.

FASTNC is quiet on import. Applications may configure the ``fastnc`` logger
hierarchy directly, while notebooks and small scripts can use the convenience
functions in this module.
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
_handler: logging.Handler | None = None


def configure_logging(
    level: int | str = logging.INFO,
    *,
    stream=None,
) -> logging.Logger:
    """Enable concise FASTNC progress logging and return its root logger.

    This convenience function is intended for notebooks and small scripts.
    Applications that already configure Python logging may instead configure
    the ``"fastnc"`` logger hierarchy directly.
    """
    global _handler
    logger = logging.getLogger("fastnc")
    resolved = (
        logging._nameToLevel.get(level.upper())
        if isinstance(level, str)
        else level
    )
    if not isinstance(resolved, int):
        raise ValueError(f"unknown logging level: {level!r}")
    logger.setLevel(resolved)
    if _handler is None:
        _handler = logging.StreamHandler(stream)
        _handler.setFormatter(logging.Formatter("%(levelname)s fastnc: %(message)s"))
        logger.addHandler(_handler)
    elif stream is not None and getattr(_handler, "stream", None) is not stream:
        _handler.setStream(stream)
    _handler.setLevel(resolved)
    return logger


def disable_logging() -> None:
    """Remove the handler installed by :func:`configure_logging`."""
    global _handler
    if _handler is not None:
        logging.getLogger("fastnc").removeHandler(_handler)
        _handler.close()
        _handler = None


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
