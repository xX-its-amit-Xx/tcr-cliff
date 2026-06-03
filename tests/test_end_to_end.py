# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""End-to-end integration on the toy set: data -> cliffs -> model -> cliff-aware eval.

These tests assert the *machinery* is sound (the pipeline composes, reports are
well-formed, signal is learnable). They deliberately do **not** assert that a
particular model wins on cliff records — that is a benchmark claim reserved for
real data; on the small synthetic toy set such an outcome would be noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from tcr_cliff.cliffs import find_neighbor_pairs
from tcr_cliff.config import CliffConfig, Config
from tcr_cliff.data import load_toy, split_pairs
from tcr_cliff.eval import cliff_aware_report, compare_models
from tcr_cliff.models import predict_scores, train_model


def _cfg(kind: str) -> Config:
    cfg = Config()
    cfg.embedding.backend = "fallback"
    cfg.embedding.fallback_dim = 96
    cfg.embedding.include_mhc = False
    cfg.embedding.cache_dir = None
    cfg.model.kind = kind
    cfg.model.baseline.n_estimators = 120
    cfg.model.cliff_aware.contrastive_epochs = 5
    cfg.model.cliff_aware.head_epochs = 10
    cfg.train.seed = 0
    return cfg


def test_full_baseline_pipeline_and_cliff_report():
    df = load_toy()
    splits = split_pairs(df)
    test = splits["test"].reset_index(drop=True)
    pairs = find_neighbor_pairs(test, CliffConfig(vary="both"))

    model = train_model(_cfg("baseline_lgbm"), splits["train"], splits["val"])
    scores = predict_scores(model, test)
    assert scores.shape == (len(test),)

    report = cliff_aware_report(test, scores, pairs=pairs, threshold=0.5)
    # Every stratum present and the overall metric finite + better than chance.
    assert set(report) == {
        "overall",
        "cliff_records",
        "non_cliff_records",
        "pair_level",
        "enhanced",
        "gap",
    }
    # Enhanced robust metrics are computed.
    assert np.isfinite(report["enhanced"]["cliff_auc"])
    assert "crg" in report["enhanced"]
    assert np.isfinite(report["overall"]["auroc"])
    assert report["overall"]["auroc"] > 0.6
    # Pair-level headline numbers are computed (cliff pairs exist in the toy test split).
    assert report["pair_level"]["n_cliff_pairs"] >= 1
    assert np.isfinite(report["pair_level"]["cliff_pair_directional_accuracy"])
    assert 0.0 <= report["pair_level"]["cliff_recovery_rate"] <= 1.0


def test_compare_models_table():
    df = load_toy()
    splits = split_pairs(df)
    test = splits["test"].reset_index(drop=True)
    pairs = find_neighbor_pairs(test, CliffConfig(vary="both"))

    model = train_model(_cfg("baseline_lgbm"), splits["train"])
    rep = cliff_aware_report(test, predict_scores(model, test), pairs=pairs)
    tbl = compare_models({"baseline": rep})
    assert "cliff_record_auroc" in tbl.columns
    assert "cliff_auc" in tbl.columns
    assert "crg" in tbl.columns
    assert len(tbl) == 1


def test_cliff_aware_pipeline_runs():
    pytest.importorskip("torch")
    df = load_toy()
    splits = split_pairs(df)
    test = splits["test"].reset_index(drop=True)

    model = train_model(_cfg("cliff_aware"), splits["train"], splits["val"])
    scores = predict_scores(model, test)
    assert scores.shape == (len(test),)
    assert np.all((scores >= 0) & (scores <= 1))
    report = cliff_aware_report(test, scores, cliff_cfg=CliffConfig(vary="both"))
    assert np.isfinite(report["overall"]["auroc"])
