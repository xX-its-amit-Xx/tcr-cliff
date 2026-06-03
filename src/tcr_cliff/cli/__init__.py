# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Command-line interface package for tcr-cliff.

Exposes the Typer ``app`` and the ``main`` console entry point so callers can do
``from tcr_cliff.cli import app`` or invoke ``tcr-cliff`` on the command line.
"""

from __future__ import annotations

from tcr_cliff.cli.main import app, main

__all__ = ["app", "main"]
