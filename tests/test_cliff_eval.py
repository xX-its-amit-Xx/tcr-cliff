# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Tests for the headline cliff-aware evaluation."""

from __future__ import annotations

import math

import numpy as np

from tcr_cliff.cliffs.detect import find_neighbor_pairs
from tcr_cliff.config import CliffConfig
from tcr_cliff.eval.cliff_eval import (
    cliff_aware_report,
    cliff_recovery_rate,
    pair_directional_accuracy,
)


def test_directional_accuracy_perfect_and_reversed(tiny_df):
    pairs = find_neighbor_pairs(tiny_df, CliffConfig(vary="both"))
    # tiny_df rows: r0 binder=1, r1 binder=0 (cliff). Correct scores rank r0 > r1.
    good = np.array([0.9, 0.1, 0.85, 0.2])  # aligned to r0..r3
    res = pair_directional_accuracy(pairs, good)
    assert math.isclose(res["cliff_pair_directional_accuracy"], 1.0)
    # Reverse the cliff members -> directional accuracy 0.
    bad = np.array([0.1, 0.9, 0.85, 0.2])
    res_bad = pair_directional_accuracy(pairs, bad)
    assert math.isclose(res_bad["cliff_pair_directional_accuracy"], 0.0)


def test_cliff_recovery_rate(tiny_df):
    pairs = find_neighbor_pairs(tiny_df, CliffConfig(vary="both"))
    perfect = np.array([0.9, 0.1, 0.9, 0.1])  # both cliff members correct at 0.5
    assert math.isclose(cliff_recovery_rate(pairs, perfect, threshold=0.5), 1.0)
    wrong = np.array([0.1, 0.1, 0.9, 0.1])  # r0 misclassified
    assert math.isclose(cliff_recovery_rate(pairs, wrong, threshold=0.5), 0.0)


def test_cliff_aware_report_structure(tiny_df):
    scores = np.array([0.9, 0.1, 0.85, 0.2])
    rep = cliff_aware_report(tiny_df, scores, cliff_cfg=CliffConfig(vary="both"))
    assert set(rep) == {
        "overall",
        "cliff_records",
        "non_cliff_records",
        "pair_level",
        "enhanced",
        "gap",
    }
    assert rep["pair_level"]["n_cliff_pairs"] >= 1
    # length mismatch must raise
    try:
        cliff_aware_report(tiny_df, scores[:2])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_cliff_gap_on_toy(toy_df):
    # A deliberately cliff-blind predictor (score depends only on peptide P2/P9 anchors,
    # ignoring that some anchor variants flip) should do worse on cliff records.
    import pandas as pd

    df = toy_df[toy_df["split"] == "test"].reset_index(drop=True)
    if len(df) < 10:
        df = toy_df.reset_index(drop=True)
    # "Leaky" near-oracle: mostly the true label, so report is well-defined and finite.
    rng = np.random.default_rng(0)
    scores = np.where(
        df["binder"] == 1, rng.uniform(0.5, 1.0, len(df)), rng.uniform(0.0, 0.5, len(df))
    )
    rep = cliff_aware_report(df, scores, cliff_cfg=CliffConfig(vary="peptide"))
    assert isinstance(rep["gap"]["auroc_gap"], float)
    assert not pd.isna(rep["overall"]["auroc"])
