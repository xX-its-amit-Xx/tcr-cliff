# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Tests for standard metric math."""

from __future__ import annotations

import math

import numpy as np

from tcr_cliff.eval.metrics import binary_metrics, bootstrap_ci, regression_metrics


def test_binary_metrics_perfect_separation():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    m = binary_metrics(y, s)
    assert math.isclose(m["auroc"], 1.0)
    assert math.isclose(m["auprc"], 1.0)
    assert math.isclose(m["accuracy"], 1.0)
    assert m["n"] == 4 and m["n_pos"] == 2


def test_binary_metrics_single_class_is_nan():
    y = np.array([1, 1, 1])
    s = np.array([0.2, 0.7, 0.9])
    m = binary_metrics(y, s)
    assert math.isnan(m["auroc"])
    assert math.isnan(m["auprc"])
    # hard-label metrics are still defined
    assert not math.isnan(m["accuracy"])


def test_binary_metrics_random_is_half():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    s = rng.random(2000)
    m = binary_metrics(y, s)
    assert 0.45 < m["auroc"] < 0.55


def test_regression_metrics_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = regression_metrics(y, y.copy())
    assert math.isclose(m["pearson"], 1.0, abs_tol=1e-9)
    assert math.isclose(m["spearman"], 1.0, abs_tol=1e-9)
    assert math.isclose(m["rmse"], 0.0, abs_tol=1e-9)


def test_regression_metrics_constant_is_nan():
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([5.0, 5.0, 5.0])
    m = regression_metrics(y, pred)
    assert math.isnan(m["pearson"])


def test_bootstrap_ci_orders():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(100), np.ones(100)].astype(int)
    s = np.r_[rng.normal(0.3, 0.1, 100), rng.normal(0.7, 0.1, 100)]
    point, lo, hi = bootstrap_ci(y, s, metric="auroc", n_boot=200, seed=0)
    assert lo <= point <= hi
    assert hi > 0.8
