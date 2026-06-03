# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Evaluation: standard metrics + the headline cliff-aware evaluation."""

from __future__ import annotations

from tcr_cliff.eval.cliff_eval import (
    cliff_aware_report,
    cliff_recovery_rate,
    pair_directional_accuracy,
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
    "cliff_aware_report",
    "cliff_recovery_rate",
    "cliff_report_to_frame",
    "compare_models",
    "pair_directional_accuracy",
    "regression_metrics",
    "save_report",
]
