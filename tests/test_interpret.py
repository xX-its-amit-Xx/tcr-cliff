# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Tests for the interpretation module (no torch / no network required).

These trains a tiny baseline on a handful of toy rows using the dependency-free
fallback embedder, then exercises occlusion attribution, SHAP field aggregation,
and anchor analysis. SHAP and matplotlib are installed in CI; the baseline falls
back to scikit-learn's HistGBM when LightGBM is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tcr_cliff.config import Config
from tcr_cliff.data import load_toy
from tcr_cliff.interpret import (
    anchor_importance,
    explain_baseline,
    map_to_positions,
    plot_field_importance,
    plot_residue_importance,
    residue_attribution,
)
from tcr_cliff.interpret.anchors import anchor_importance as anchor_importance_direct
from tcr_cliff.models.baseline import BaselineModel


def _small_config(tmp_path) -> Config:
    """Build a tiny, offline-friendly config (fallback embedder, no disk cache)."""
    return Config.model_validate(
        {
            "seed": 0,
            "output_dir": str(tmp_path / "out"),
            "embedding": {
                "backend": "fallback",
                "fallback_dim": 16,
                "include_mhc": True,
                "cache_dir": None,
            },
            "model": {
                "kind": "baseline_lgbm",
                "baseline": {"n_estimators": 25, "num_leaves": 7, "prefer": "sklearn"},
            },
        }
    )


@pytest.fixture(scope="module")
def toy_df() -> pd.DataFrame:
    """A small slice of the bundled synthetic toy set."""
    df = load_toy().head(24).reset_index(drop=True)
    # Guarantee both classes are present for a trainable baseline.
    assert df["binder"].nunique() == 2
    return df


@pytest.fixture()
def fitted_baseline(toy_df, tmp_path) -> BaselineModel:
    """A tiny baseline fitted on the toy slice."""
    cfg = _small_config(tmp_path)
    model = BaselineModel(cfg).fit(toy_df)
    return model


def test_residue_attribution_length_and_finite(fitted_baseline, toy_df):
    """Occlusion attribution returns one finite value per peptide residue."""
    row = toy_df.iloc[0]
    peptide = row["peptide"]

    def predict_fn(df: pd.DataFrame) -> np.ndarray:
        return fitted_baseline.predict_proba(df)

    attr = residue_attribution(predict_fn, row, field="peptide")
    assert isinstance(attr, np.ndarray)
    assert attr.shape == (len(peptide),)
    assert np.all(np.isfinite(attr))
    assert np.isfinite(np.max(np.abs(attr)))


def test_residue_attribution_cdr3b_field(fitted_baseline, toy_df):
    """Attribution also works on a non-default sequence field."""
    row = toy_df.iloc[1]

    def predict_fn(df: pd.DataFrame) -> np.ndarray:
        return fitted_baseline.predict_proba(df)

    attr = residue_attribution(predict_fn, row, field="cdr3b")
    assert attr.shape == (len(row["cdr3b"]),)
    assert np.all(np.isfinite(attr))


def test_explain_baseline_per_field_keys(fitted_baseline, toy_df):
    """SHAP aggregation produces per-field importance blocks."""
    out = explain_baseline(fitted_baseline, toy_df, max_display=10)
    assert "field_importance" in out
    fields = set(out["field_importance"])
    # Embedded fields are cdr3b, peptide, mhc_pseudo (include_mhc=True).
    assert {"cdr3b", "peptide", "mhc_pseudo"} <= fields
    assert all(v >= 0.0 for v in out["field_importance"].values())
    assert len(out["top_features"]) <= 10
    assert out["n_samples"] == len(toy_df)
    assert out["mean_abs_shap"].shape[0] == len(out["feature_names"])


def test_anchor_importance_structure():
    """anchor_importance reports P2, POmega and a dominance flag."""
    peptide = "GILGFVFTL"
    attr = np.array([0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8])
    out = anchor_importance(attr, peptide)
    assert set(out["anchors"]) == {"P2", "POmega"}
    assert out["anchors"]["P2"]["position"] == 2
    assert out["anchors"]["P2"]["residue"] == "I"
    assert out["anchors"]["POmega"]["position"] == len(peptide)
    assert out["anchors"]["POmega"]["residue"] == "L"
    assert 0.0 <= out["anchor_fraction"] <= 1.0
    assert out["anchors_dominate"] is True
    assert out["length"] == len(peptide)
    # Same function exported at package and module level.
    assert anchor_importance_direct(attr, peptide) == out


def test_anchor_importance_length_mismatch():
    """Mismatched attribution length raises ValueError."""
    with pytest.raises(ValueError):
        anchor_importance(np.zeros(3), "GILGFVFTL")


def test_map_to_positions():
    """map_to_positions zips 1-based positions, residues and importances."""
    attr = np.array([0.2, -0.1, 0.3])
    mapped = map_to_positions(attr, "ABC")
    assert mapped[0] == (1, "A", pytest.approx(0.2))
    assert mapped[1] == (2, "B", pytest.approx(-0.1))
    assert mapped[2] == (3, "C", pytest.approx(0.3))


def test_attribution_to_anchor_pipeline(fitted_baseline, toy_df):
    """End-to-end: occlusion -> anchor analysis on a real peptide."""
    row = toy_df.iloc[0]
    peptide = row["peptide"]

    def predict_fn(df: pd.DataFrame) -> np.ndarray:
        return fitted_baseline.predict_proba(df)

    attr = residue_attribution(predict_fn, row, field="peptide")
    out = anchor_importance(attr, peptide)
    assert out["length"] == len(peptide)
    assert isinstance(out["anchors_dominate"], bool)


def test_plots_write_png(fitted_baseline, toy_df, tmp_path):
    """Plot helpers save PNGs without ever calling plt.show()."""
    row = toy_df.iloc[0]
    peptide = row["peptide"]

    def predict_fn(df: pd.DataFrame) -> np.ndarray:
        return fitted_baseline.predict_proba(df)

    attr = residue_attribution(predict_fn, row, field="peptide")
    p1 = plot_residue_importance(attr, peptide, tmp_path / "res.png")
    assert p1.exists() and p1.stat().st_size > 0

    expl = explain_baseline(fitted_baseline, toy_df, max_display=5)
    p2 = plot_field_importance(expl["field_importance"], tmp_path / "field.png")
    assert p2.exists() and p2.stat().st_size > 0
