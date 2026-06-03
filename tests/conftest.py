# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from tcr_cliff.data import load_toy


@pytest.fixture(scope="session")
def toy_df() -> pd.DataFrame:
    """The bundled synthetic toy pairs table (validated)."""
    return load_toy()


@pytest.fixture()
def tiny_df() -> pd.DataFrame:
    """A hand-built frame with one known cliff and one known smooth neighbour.

    Rows (all share cdr3b + MHC, so they are peptide-neighbours):
      r0  GILGFVFTL  binder=1  (anchor)
      r1  GILGFVFTA  binder=0  -> Levenshtein 1 from r0, label flip => CLIFF
      r2  GILAFVFTL  binder=1  -> Levenshtein 1 from r0, same label  => SMOOTH
      r3  (different cdr3b, same peptide as r0) -> NOT a peptide neighbour (identical
          peptide) and NOT a cdr3b neighbour (far cdr3b): isolated control.
    """
    return pd.DataFrame(
        {
            "pair_id": ["r0", "r1", "r2", "r3"],
            "cdr3b": [
                "CASSLAPGATNEKLFF",
                "CASSLAPGATNEKLFF",
                "CASSLAPGATNEKLFF",
                "CASRPDREYTQYF",
            ],
            "peptide": ["GILGFVFTL", "GILGFVFTA", "GILAFVFTL", "GILGFVFTL"],
            "mhc": ["HLA-A*02:01"] * 4,
            "mhc_pseudo": ["YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY"] * 4,
            "binder": [1, 0, 1, 0],
            "affinity": [8.0, 4.0, 7.5, 3.8],
            "source": ["test"] * 4,
            "split": ["test"] * 4,
        }
    )
