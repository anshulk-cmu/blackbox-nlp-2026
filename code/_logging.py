"""Shared logging utility for Phase 1 + downstream code.

Mirrors the convention from toy/run_toy.py: emit to both stdout and a
per-run file under $BLACKBOX_DATA/logs/. Idempotent — re-calling with
the same logger name returns the existing logger without duplicating
handlers.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import sys
from typing import Optional

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str,
    log_path: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a logger writing to stdout and (optionally) a file.

    Args:
      name:     Logger name; usually __name__ from the caller.
      log_path: If given, append to this file. Parent dir is auto-created.
      level:    Log level (default INFO).
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def default_log_path(data_dir: str, phase_tag: str) -> str:
    """Standard log path: $BLACKBOX_DATA/logs/{phase_tag}_{TS}.log."""
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(data_dir, "logs", f"{phase_tag}_{ts}.log")
