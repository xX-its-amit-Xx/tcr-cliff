# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Render evaluation results to JSON and human-readable tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tcr_cliff._logging import get_logger

_log = get_logger("eval.report")


def cliff_report_to_frame(report: dict) -> pd.DataFrame:
    """Flatten a :func:`cliff_aware_report` result into a tidy comparison table."""
    rows = []
    for stratum in ("overall", "cliff_records", "non_cliff_records"):
        m = report.get(stratum, {})
        rows.append(
            {
                "stratum": stratum,
                "n": m.get("n"),
                "auroc": m.get("auroc"),
                "auprc": m.get("auprc"),
                "accuracy": m.get("accuracy"),
                "f1": m.get("f1"),
            }
        )
    return pd.DataFrame(rows)


def compare_models(reports: dict[str, dict]) -> pd.DataFrame:
    """Build a benchmark table comparing several models' cliff-aware results.

    Parameters
    ----------
    reports:
        ``{model_name: cliff_aware_report(...)}``.

    Returns
    -------
    DataFrame
        One row per model with overall AUROC, cliff-record AUROC, the gap, and the
        cliff-pair directional accuracy — i.e. the headline comparison.
    """
    rows = []
    for name, rep in reports.items():
        rows.append(
            {
                "model": name,
                "overall_auroc": rep["overall"].get("auroc"),
                "cliff_record_auroc": rep["cliff_records"].get("auroc"),
                "non_cliff_record_auroc": rep["non_cliff_records"].get("auroc"),
                "auroc_gap": rep["gap"].get("auroc_gap"),
                "cliff_pair_dir_acc": rep["pair_level"].get("cliff_pair_directional_accuracy"),
                "cliff_recovery_rate": rep["pair_level"].get("cliff_recovery_rate"),
            }
        )
    return pd.DataFrame(rows)


def save_report(report: dict, path: str | Path) -> Path:
    """Write an evaluation report to JSON (creating parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=float))
    _log.info("wrote eval report to %s", path)
    return path
