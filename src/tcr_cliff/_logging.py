# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Central logging configuration.

The package never uses ``print``; all user-facing progress goes through the
standard :mod:`logging` library so callers can control verbosity and routing.
"""

from __future__ import annotations

import logging
import os
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_CONFIGURED = False


def setup_logging(level: int | str | None = None, *, fmt: str = _DEFAULT_FORMAT) -> None:
    """Configure the root ``tcr_cliff`` logger once.

    Parameters
    ----------
    level:
        Logging level (``logging.INFO`` by default, or the ``TCR_CLIFF_LOGLEVEL``
        environment variable if set). Accepts ints or level names.
    fmt:
        Log line format string.
    """
    global _CONFIGURED
    if level is None:
        level = os.environ.get("TCR_CLIFF_LOGLEVEL", "INFO")
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())

    logger = logging.getLogger("tcr_cliff")
    logger.setLevel(level)
    # Avoid duplicate handlers when called repeatedly (e.g. from notebooks).
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.propagate = False
    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a namespaced child logger, configuring logging on first use.

    Parameters
    ----------
    name:
        Sub-logger name. ``get_logger("cliffs")`` yields ``tcr_cliff.cliffs``.
    """
    if not _CONFIGURED:
        setup_logging()
    base = logging.getLogger("tcr_cliff")
    return base if not name else base.getChild(name)
