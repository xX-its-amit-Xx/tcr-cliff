# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Tests for structural-confidence parsing and feature assembly.

Biopython is installed in the test environment, so these run without skips.
"""

from __future__ import annotations

import numpy as np

from tcr_cliff.config import StructureConfig
from tcr_cliff.data import load_toy, toy_path
from tcr_cliff.structure import (
    STRUCTURE_FEATURE_NAMES,
    ConfidenceRecord,
    load_structure_features,
    parse_af_json,
    parse_pdb_plddt,
)
from tcr_cliff.structure.features import (
    features_from_record,
    features_from_summary,
    missing_vector,
)


def _example_json(pair_id: str):
    return toy_path(f"structure/examples/{pair_id}.json")


def test_feature_names_length_is_twelve():
    assert len(STRUCTURE_FEATURE_NAMES) == 12
    # The first eight mirror the summary schema, in order.
    assert STRUCTURE_FEATURE_NAMES[:8] == [
        "ptm",
        "iptm",
        "plddt_mean",
        "plddt_min",
        "pae_interface_mean",
        "pae_interface_min",
        "n_interface_contacts",
        "plddt_interface_mean",
    ]
    assert STRUCTURE_FEATURE_NAMES[-1] == "available_flag"


def test_parse_af_json_returns_record():
    rec = parse_af_json(_example_json("toy0001"))
    assert isinstance(rec, ConfidenceRecord)
    assert rec.ptm == 0.7802
    assert rec.iptm == 0.739
    assert rec.plddt.ndim == 1
    assert rec.pae.shape == (rec.n_tokens, rec.n_tokens)
    assert rec.tcr_chain == "A"
    assert "B" in rec.pmhc_chains
    assert set(rec.token_chain_ids) == {"A", "B"}


def test_features_from_record_is_twelve_dim_and_finite():
    rec = parse_af_json(_example_json("toy0001"))
    feats = features_from_record(rec, cutoff=8.0)
    assert feats.shape == (12,)
    assert np.isfinite(feats).all()
    # ptm/iptm land in the first two slots.
    assert feats[0] == 0.7802
    assert feats[1] == 0.739
    # available_flag is set for a real record.
    assert feats[-1] == 1.0


def test_binder_example_has_higher_iptm_than_non_binder():
    # toy0001 is a binder (iptm 0.739); toy0000 is a non-binder (iptm 0.2809).
    binder = parse_af_json(_example_json("toy0001"))
    non_binder = parse_af_json(_example_json("toy0000"))
    assert binder.iptm > non_binder.iptm
    # And that ordering is preserved in the feature vector (iptm is column 1).
    fb = features_from_record(binder)
    fn = features_from_record(non_binder)
    assert fb[1] > fn[1]


def test_parse_pdb_plddt_reads_ca_bfactors():
    out = parse_pdb_plddt(toy_path("structure/examples/example.pdb"))
    plddt = out["plddt"]
    chain_ids = out["chain_ids"]
    assert isinstance(plddt, np.ndarray)
    assert len(plddt) == len(chain_ids)
    assert len(plddt) > 0
    # B-factor column holds pLDDT in [0, 100]; first CA is 81.55.
    assert np.isclose(plddt[0], 81.55)
    assert set(chain_ids) == {"A", "B"}


def test_features_from_summary_matches_manual():
    summary = {
        "ptm": 0.8,
        "iptm": 0.7,
        "plddt_mean": 82.0,
        "plddt_min": 65.0,
        "pae_interface_mean": 6.0,
        "pae_interface_min": 2.0,
        "n_interface_contacts": 100,
        "plddt_interface_mean": 83.0,
    }
    feats = features_from_summary(summary)
    assert feats.shape == (12,)
    assert np.isclose(feats[8], 0.7 * 0.8)  # iptm_x_ptm
    assert np.isclose(feats[9], 100 / 100.0)  # contact_density
    assert np.isclose(feats[10], 6.0 - 2.0)  # pae_interface_range
    assert feats[11] == 1.0  # available_flag


def test_missing_vector_is_unavailable_zeros():
    mv = missing_vector()
    assert mv.shape == (12,)
    assert np.allclose(mv, 0.0)
    assert mv[-1] == 0.0
    assert np.allclose(features_from_summary(None), mv)
    assert np.allclose(features_from_record(None), mv)


def test_load_structure_features_shape_on_toy():
    df = load_toy()
    X, names = load_structure_features(df, StructureConfig())
    assert X.shape == (len(df), 12)
    assert names == STRUCTURE_FEATURE_NAMES
    assert np.isfinite(X).all()
    # The toy summary covers every pair, so all rows should be available.
    available = X[:, -1]
    assert available.mean() > 0.99


def test_load_structure_features_missing_rows_get_zeros():
    df = load_toy().head(3).copy()
    df["pair_id"] = ["does-not-exist-0", "does-not-exist-1", "does-not-exist-2"]
    X, _ = load_structure_features(df, StructureConfig())
    assert X.shape == (3, 12)
    assert np.allclose(X, 0.0)
