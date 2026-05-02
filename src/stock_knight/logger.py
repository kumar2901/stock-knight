"""
Centralised logging configuration for stock-knight.

Usage anywhere in the package:
    from stock_knight.logger import get_logger
    log = get_logger(__name__)
    log.info("message")
"""
from __future__ import annotations

import logging
import os
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring the root handler on first call."""
    logger = logging.getLogger(name)
    if not logging.root.handlers:
        _configure_root()
    return logger


def _configure_root() -> None:
    level_name = os.environ.get("STOCK_KNIGHT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logging.root.setLevel(level)
    logging.root.addHandler(handler)
