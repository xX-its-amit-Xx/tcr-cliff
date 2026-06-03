# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Evaluation: standard metrics + the headline cliff-aware evaluation."""

from __future__ import annotations

from tcr_cliff.eval.cliff_eval import (
    cliff_aware_report,
    cliff_recovery_rate,
    cross_validated_cliff_eval,
    pair_directional_accuracy,
)
from tcr_cliff.eval.cliff_metrics import (
    bootstrap_cliff_auc,
    cliff_auc,
    cliff_responsiveness_gap,
    enhanced_cliff_metrics,
    evaluate_pairs,
    nearest_neighbor_scores,
    pair_steepness,
)
from tcr_cliff.eval.metrics import binary_metrics, bootstrap_ci, regression_metrics
from tcr_cliff.eval.report import (
    cliff_report_to_frame,
    compare_models,
    save_report,
)

__all__ = [
    "binary_metrics",
    "bootstrap_ci",
    "bootstrap_cliff_auc",
    "cliff_auc",
    "cliff_aware_report",
    "cliff_recovery_rate",
    "cliff_report_to_frame",
    "cliff_responsiveness_gap",
    "compare_models",
    "cross_validated_cliff_eval",
    "enhanced_cliff_metrics",
    "evaluate_pairs",
    "nearest_neighbor_scores",
    "pair_directional_accuracy",
    "pair_steepness",
    "regression_metrics",
    "save_report",
]
