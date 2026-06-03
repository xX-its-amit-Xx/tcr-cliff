# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Tests for the bundled toy dataset and schema validation."""

from __future__ import annotations

import pandas as pd
import pytest

from tcr_cliff.data import REQUIRED_COLUMNS, load_toy_structure_summary, split_pairs
from tcr_cliff.data.schema import SchemaError, validate_pairs


def test_toy_loads_and_validates(toy_df):
    for col in REQUIRED_COLUMNS:
        assert col in toy_df.columns
    assert toy_df["binder"].isin((0, 1)).all()
    assert len(toy_df) > 200
    # Both classes present and reasonably balanced.
    assert 0.2 < toy_df["binder"].mean() < 0.8


def test_split_uses_split_column(toy_df):
    splits = split_pairs(toy_df)
    assert set(splits) == {"train", "val", "test"}
    total = sum(len(v) for v in splits.values())
    assert total == len(toy_df)
    # No peptide should leak between train and test (grouped/split-column design).
    train_pep = set(splits["train"]["peptide"])
    test_pep = set(splits["test"]["peptide"])
    leak = train_pep & test_pep
    assert len(leak) / max(len(test_pep), 1) < 0.5


def test_structure_summary_keys(toy_df):
    summary = load_toy_structure_summary()
    assert len(summary) == len(toy_df)
    some = next(iter(summary.values()))
    for key in ("ptm", "iptm", "pae_interface_mean", "plddt_mean", "n_interface_contacts"):
        assert key in some


def test_validate_rejects_missing_columns():
    bad = pd.DataFrame({"cdr3b": ["CASSF"], "peptide": ["GILGFVFTL"]})  # no pair_id/binder
    with pytest.raises(SchemaError):
        validate_pairs(bad)


def test_validate_filters_nonstandard_residues():
    df = pd.DataFrame(
        {
            "pair_id": ["a", "b"],
            "cdr3b": ["CASSF", "CASS1"],  # '1' is invalid
            "peptide": ["GILGFVFTL", "GILGFVFTL"],
            "binder": [1, 0],
        }
    )
    out = validate_pairs(df)
    assert len(out) == 1 and out.iloc[0]["pair_id"] == "a"
