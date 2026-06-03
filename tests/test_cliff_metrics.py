# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Tests for the enhanced, literature-grounded cliff metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from tcr_cliff.cliffs.detect import NeighborPair
from tcr_cliff.eval.cliff_metrics import (
    bootstrap_cliff_auc,
    cliff_auc,
    cliff_responsiveness_gap,
    enhanced_cliff_metrics,
    evaluate_pairs,
    nearest_neighbor_scores,
    pair_steepness,
)


def _pair(i, j, li, lj, is_cliff, dist=1):
    return NeighborPair(
        i=i,
        j=j,
        id_i=f"r{i}",
        id_j=f"r{j}",
        vary="cdr3b",
        seq_i="A",
        seq_j="B",
        distance=dist,
        label_i=li,
        label_j=lj,
        affinity_i=float("nan"),
        affinity_j=float("nan"),
        is_cliff=is_cliff,
        reason="label_flip" if is_cliff else "smooth",
    )


def test_cliff_auc_perfect_and_reversed():
    pairs = [_pair(0, 1, 1, 0, True), _pair(2, 3, 0, 1, True)]
    # binder of pair0 is idx0; binder of pair1 is idx3. Rank binders high.
    good = np.array([0.9, 0.1, 0.1, 0.9])
    assert math.isclose(cliff_auc(evaluate_pairs(pairs, good)), 1.0)
    bad = np.array([0.1, 0.9, 0.9, 0.1])
    assert math.isclose(cliff_auc(evaluate_pairs(pairs, bad)), 0.0)


def test_sali_weighting_emphasises_steep_cliffs():
    # Steep (distance-1) cliff ranked CORRECTLY; shallow (distance-2) cliff ranked WRONG.
    pairs = [_pair(0, 1, 1, 0, True, dist=1), _pair(2, 3, 1, 0, True, dist=2)]
    scores = np.array([0.9, 0.1, 0.1, 0.9])  # pair0 correct, pair1 wrong
    ev = evaluate_pairs(pairs, scores)
    assert math.isclose(cliff_auc(ev, weighted=False), 0.5)  # 1 of 2 correct
    # SALI weights: 1.0 for dist-1, 0.5 for dist-2 -> (1*1 + 0.5*0)/1.5
    assert math.isclose(cliff_auc(ev, weighted=True), 1.0 / 1.5, rel_tol=1e-9)


def test_pair_steepness():
    assert pair_steepness(_pair(0, 1, 1, 0, True, dist=1)) == 1.0
    assert pair_steepness(_pair(0, 1, 1, 0, True, dist=2)) == 0.5
    assert pair_steepness(_pair(0, 1, 1, 1, False, dist=1)) == 0.0  # no activity change


def test_cliff_responsiveness_gap():
    # Cliff pairs: big prediction change; smooth pairs: tiny change -> CRG > 0.
    pairs = [_pair(0, 1, 1, 0, True), _pair(2, 3, 1, 1, False)]
    scores = np.array([0.95, 0.05, 0.50, 0.52])  # cliff |Δ|=0.9, smooth |Δ|=0.02
    crg = cliff_responsiveness_gap(evaluate_pairs(pairs, scores))
    assert crg["crg"] > 0.5
    assert crg["mean_cliff_delta"] > crg["mean_smooth_delta"]
    # A perfectly smooth model (identical neighbour predictions) -> CRG ~ 0.
    flat = np.array([0.5, 0.5, 0.5, 0.5])
    assert math.isclose(cliff_responsiveness_gap(evaluate_pairs(pairs, flat))["crg"], 0.0)


def test_nearest_neighbor_baseline_fails_on_cliff():
    # Train holds a cliff: two CDR3b one edit apart, same peptide, opposite labels.
    train = pd.DataFrame(
        {
            "pair_id": ["a", "b"],
            "cdr3b": ["CASSLF", "CASSYF"],
            "peptide": ["GILGFVFTL", "GILGFVFTL"],
            "binder": [1, 0],
        }
    )
    query = pd.DataFrame(
        {
            "pair_id": ["q1", "q2", "q3"],
            "cdr3b": ["CASSLF", "CASSYF", "CASSWF"],  # exact, exact, equidistant
            "peptide": ["GILGFVFTL"] * 3,
            "binder": [1, 0, 1],
        }
    )
    s = nearest_neighbor_scores(train, query)
    assert math.isclose(s[0], 1.0)  # exact match to the binder
    assert math.isclose(s[1], 0.0)  # exact match to the non-binder
    # Equidistant from both cliff members -> the baseline cannot resolve it (0.5).
    assert math.isclose(s[2], 0.5)


def test_bootstrap_and_enhanced_block():
    pairs = [_pair(2 * k, 2 * k + 1, 1, 0, True) for k in range(8)]
    rng = np.random.default_rng(0)
    scores = np.zeros(16)
    scores[0::2] = rng.uniform(0.6, 1.0, 8)  # binders high
    scores[1::2] = rng.uniform(0.0, 0.4, 8)  # non-binders low
    ev = evaluate_pairs(pairs, scores)
    point, lo, hi = bootstrap_cliff_auc(ev, n_boot=200, seed=0)
    assert math.isclose(point, 1.0)
    assert lo <= point <= hi + 1e-9
    block = enhanced_cliff_metrics(pairs, scores, n_boot=100)
    for key in ("cliff_auc", "sali_weighted_cliff_auc", "crg", "crg_effect_size", "cliff_auc_lo"):
        assert key in block
