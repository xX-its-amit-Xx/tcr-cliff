# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Tests for the gradient-boosted baseline and the model registry."""

from __future__ import annotations

import numpy as np

from tcr_cliff.config import Config
from tcr_cliff.data import split_pairs
from tcr_cliff.eval.metrics import binary_metrics
from tcr_cliff.models import load_model, train_model


def _fast_cfg(tmp_path) -> Config:
    cfg = Config()
    cfg.embedding.backend = "fallback"
    cfg.embedding.fallback_dim = 96
    cfg.embedding.include_mhc = False
    cfg.embedding.cache_dir = None
    cfg.model.kind = "baseline_lgbm"
    cfg.model.baseline.n_estimators = 60
    cfg.output_dir = tmp_path
    return cfg


def test_baseline_trains_and_learns_signal(toy_df, tmp_path):
    cfg = _fast_cfg(tmp_path)
    splits = split_pairs(toy_df)
    model = train_model(cfg, splits["train"], splits["val"])
    scores = model.predict_proba(splits["test"])
    assert scores.shape == (len(splits["test"]),)
    assert np.all((scores >= 0) & (scores <= 1))
    # The toy set has a learnable binding rule, so the baseline should beat chance.
    m = binary_metrics(splits["test"]["binder"].to_numpy(), scores)
    assert m["auroc"] > 0.6


def test_baseline_save_load_roundtrip(toy_df, tmp_path):
    cfg = _fast_cfg(tmp_path)
    splits = split_pairs(toy_df)
    model = train_model(cfg, splits["train"])
    path = model.save(tmp_path / "model")
    reloaded = load_model(path)
    s1 = model.predict_proba(splits["test"])
    s2 = reloaded.predict_proba(splits["test"])
    assert np.allclose(s1, s2, atol=1e-5)
    assert reloaded.kind == "baseline_lgbm"
