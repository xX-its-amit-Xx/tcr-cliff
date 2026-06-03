# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Dataset loading: canonical schema, bundled toy set, and real-source downloaders."""

from __future__ import annotations

from tcr_cliff.data.loaders import (
    load_csv,
    load_pairs,
    load_toy,
    load_toy_structure_summary,
    split_pairs,
    toy_path,
)
from tcr_cliff.data.schema import (
    ALL_COLUMNS,
    REQUIRED_COLUMNS,
    SchemaError,
    validate_pairs,
)

__all__ = [
    "ALL_COLUMNS",
    "REQUIRED_COLUMNS",
    "SchemaError",
    "load_csv",
    "load_pairs",
    "load_toy",
    "load_toy_structure_summary",
    "split_pairs",
    "toy_path",
    "validate_pairs",
]
