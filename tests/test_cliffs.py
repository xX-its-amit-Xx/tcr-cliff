# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Tests for distance metrics and cliff detection."""

from __future__ import annotations

import itertools

from tcr_cliff.cliffs import (
    bounded_levenshtein,
    build_cliff_graph,
    cliff_hubs,
    cliff_statistics,
    find_cliffs,
    find_neighbor_pairs,
    hamming,
    levenshtein,
    within_k,
)
from tcr_cliff.cliffs.detect import _candidate_pairs_k1_levenshtein
from tcr_cliff.config import CliffConfig


def test_levenshtein_basic():
    assert levenshtein("ABC", "ABC") == 0
    assert levenshtein("ABC", "ABD") == 1  # substitution
    assert levenshtein("ABC", "AC") == 1  # deletion
    assert levenshtein("AC", "ABC") == 1  # insertion
    assert levenshtein("kitten", "sitting") == 3


def test_hamming_and_bounded():
    assert hamming("ABC", "ABD") == 1
    assert hamming("ABC", "AB") is None
    assert bounded_levenshtein("ABCDE", "ABCDE", 1) == 0
    assert bounded_levenshtein("ABCDE", "ABXDE", 1) == 1
    # distance 2 exceeds cap of 1 -> sentinel max_k+1
    assert bounded_levenshtein("ABCDE", "AXXDE", 1) == 2
    assert bounded_levenshtein("ABCDE", "VWXYZ", 1) == 2  # length-ok but far


def test_within_k():
    ok, d = within_k("GILGFVFTL", "GILGFVFTA", 1, "levenshtein")
    assert ok and d == 1
    ok, d = within_k("GILGFVFTL", "GILGFVFTA", 1, "hamming")
    assert ok and d == 1
    ok, _ = within_k("GILGFVFTL", "AAAAAAAAA", 1, "hamming")
    assert not ok


def test_deletion_index_matches_bruteforce():
    # The k=1 fast path must agree with O(n^2) brute force.
    seqs = ["GILGFVFTL", "GILGFVFTA", "GILAFVFTL", "AILGFVFTL", "QQQQQQQQQ", "GILGFVFT"]
    fast = {
        tuple(sorted(p))
        for p in _candidate_pairs_k1_levenshtein(seqs)
        if bounded_levenshtein(seqs[p[0]], seqs[p[1]], 1) <= 1
    }
    brute = {
        (i, j)
        for i, j in itertools.combinations(range(len(seqs)), 2)
        if bounded_levenshtein(seqs[i], seqs[j], 1) <= 1
    }
    assert fast == brute


def test_find_cliffs_tiny(tiny_df):
    pairs = find_neighbor_pairs(tiny_df, CliffConfig(vary="both", same_context=True))
    cliffs = [p for p in pairs if p.is_cliff]
    smooth = [p for p in pairs if not p.is_cliff]
    # r0-r1 is the label-flip cliff; r0-r2 is the smooth neighbour.
    cliff_ids = {tuple(sorted((p.id_i, p.id_j))) for p in cliffs}
    smooth_ids = {tuple(sorted((p.id_i, p.id_j))) for p in smooth}
    assert ("r0", "r1") in cliff_ids
    assert ("r0", "r2") in smooth_ids
    # r3 has a different cdr3b and an identical peptide -> participates in no pair.
    involved = {p.id_i for p in pairs} | {p.id_j for p in pairs}
    assert "r3" not in involved


def test_same_context_blocks_cross_context(tiny_df):
    # With same_context, r0 and r3 (same peptide, different cdr3b) are NOT neighbours.
    pairs = find_neighbor_pairs(tiny_df, CliffConfig(vary="peptide", same_context=True))
    ids = {tuple(sorted((p.id_i, p.id_j))) for p in pairs}
    assert ("r0", "r3") not in ids


def test_affinity_delta_criterion():
    import pandas as pd

    df = pd.DataFrame(
        {
            "pair_id": ["a", "b"],
            "cdr3b": ["CASSF", "CASSF"],
            "peptide": ["GILGFVFTL", "GILGFVFTA"],
            "mhc": ["X", "X"],
            "binder": [1, 1],  # same label -> only affinity can make it a cliff
            "affinity": [9.0, 4.0],
        }
    )
    # Without affinity criterion and same label -> not a cliff.
    cfg_label = CliffConfig(vary="peptide", require_label_flip=True, affinity_delta=None)
    assert len(find_cliffs(df, cfg_label)) == 0
    # With affinity_delta=2.0 and |9-4|=5 -> cliff.
    cfg_aff = CliffConfig(vary="peptide", require_label_flip=False, affinity_delta=2.0)
    cliffs = find_cliffs(df, cfg_aff)
    assert len(cliffs) == 1
    assert "affinity_delta" in cliffs[0].reason


def test_cliff_statistics_and_graph(toy_df):
    pairs = find_neighbor_pairs(toy_df, CliffConfig(vary="peptide"))
    stats = cliff_statistics(pairs, n_records=len(toy_df))
    assert stats["n_neighbor_pairs"] >= stats["n_cliff_pairs"] > 0
    assert 0.0 <= stats["cliff_ratio"] <= 1.0
    g = build_cliff_graph(pairs, toy_df)
    assert g.number_of_nodes() == len(toy_df)
    hubs = cliff_hubs(g, top_n=5)
    assert len(hubs) <= 5
