# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""The headline evaluation: model performance *specifically on cliff pairs*.

Most predictors look strong on aggregate metrics yet collapse on the single-residue
flips that matter clinically. This module reports two complementary views:

1. **Record-level stratification** - standard metrics computed separately on records
   that participate in a cliff vs. records that do not. The gap is the headline.
2. **Pair-level directional accuracy** - for each cliff pair (opposite outcomes,
   within ``k`` edits) does the model assign the higher score to the true binder?
   Random is 0.5. Smooth (same-outcome) neighbours provide a control.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tcr_cliff._logging import get_logger
from tcr_cliff.cliffs.detect import NeighborPair, find_neighbor_pairs, mark_cliff_membership
from tcr_cliff.config import CliffConfig
from tcr_cliff.eval.metrics import binary_metrics

_log = get_logger("eval.cliff_eval")


def pair_directional_accuracy(
    pairs: list[NeighborPair], y_score: np.ndarray, *, only_label_flip: bool = True
) -> dict[str, float]:
    """Fraction of pairs whose two members are scored in the correct order.

    For a pair with a label flip, "correct order" means the true binder receives the
    higher predicted score. Ties count as 0.5. Reported separately for cliff and
    smooth pairs.
    """
    cliff_correct = cliff_total = 0.0
    smooth_correct = smooth_total = 0.0
    for p in pairs:
        # Only label-flip pairs have a meaningful direction; skip same-label cliffs.
        if only_label_flip and p.label_i == p.label_j and p.is_cliff:
            continue
        si, sj = float(y_score[p.i]), float(y_score[p.j])
        if p.label_i == p.label_j:
            # Same-label smooth pair: "consistent" if both on the same side of 0.5-ish;
            # we score directional agreement of the *ranking* as undefined -> use 0.5.
            score = 0.5
        else:
            pos_score = si if p.label_i == 1 else sj
            neg_score = sj if p.label_i == 1 else si
            score = 1.0 if pos_score > neg_score else (0.5 if pos_score == neg_score else 0.0)
        if p.is_cliff:
            cliff_correct += score
            cliff_total += 1
        else:
            smooth_correct += score
            smooth_total += 1
    return {
        "cliff_pair_directional_accuracy": (
            (cliff_correct / cliff_total) if cliff_total else float("nan")
        ),
        "n_cliff_pairs": cliff_total,
        "smooth_pair_directional_accuracy": (
            (smooth_correct / smooth_total) if smooth_total else float("nan")
        ),
        "n_smooth_pairs": smooth_total,
    }


def cliff_recovery_rate(
    pairs: list[NeighborPair], y_score: np.ndarray, *, threshold: float = 0.5
) -> float:
    """Fraction of label-flip cliff pairs where *both* members are classified correctly.

    This is the strictest headline number: the model must get the binder AND the
    non-binder of the cliff right at the operating threshold.
    """
    correct = total = 0
    for p in pairs:
        if not p.is_cliff or p.label_i == p.label_j:
            continue
        total += 1
        pred_i = int(float(y_score[p.i]) >= threshold)
        pred_j = int(float(y_score[p.j]) >= threshold)
        if pred_i == p.label_i and pred_j == p.label_j:
            correct += 1
    return (correct / total) if total else float("nan")


def cliff_aware_report(
    df: pd.DataFrame,
    y_score: np.ndarray,
    *,
    pairs: list[NeighborPair] | None = None,
    cliff_cfg: CliffConfig | None = None,
    threshold: float = 0.5,
) -> dict:
    """Full cliff-aware evaluation for one model's predictions.

    Parameters
    ----------
    df:
        The *test* pairs table. ``y_score`` and ``pairs`` must be aligned to its
        positional row index.
    y_score:
        Predicted positive-class scores, one per row of ``df``.
    pairs:
        Pre-computed neighbour pairs on ``df``; if ``None`` they are detected here.
    cliff_cfg:
        Cliff definition used when ``pairs`` is None.
    threshold:
        Decision threshold for record-level and recovery metrics.

    Returns
    -------
    dict
        ``{"overall", "cliff_records", "non_cliff_records", "pair_level", "gap"}``.
    """
    y_score = np.asarray(y_score, dtype=float)
    if len(y_score) != len(df):
        raise ValueError(f"y_score length {len(y_score)} != n_rows {len(df)}")
    if pairs is None:
        pairs = find_neighbor_pairs(df, cliff_cfg or CliffConfig())

    df_marked = mark_cliff_membership(df.reset_index(drop=True), pairs)
    y_true = df_marked["binder"].to_numpy()
    in_cliff = df_marked["in_cliff"].to_numpy()

    overall = binary_metrics(y_true, y_score, threshold=threshold)
    cliff_m = binary_metrics(y_true[in_cliff], y_score[in_cliff], threshold=threshold)
    non_m = binary_metrics(y_true[~in_cliff], y_score[~in_cliff], threshold=threshold)

    pair_level = pair_directional_accuracy(pairs, y_score)
    pair_level["cliff_recovery_rate"] = cliff_recovery_rate(pairs, y_score, threshold=threshold)

    gap = {
        "auroc_gap": _safe_sub(non_m.get("auroc"), cliff_m.get("auroc")),
        "accuracy_gap": _safe_sub(non_m.get("accuracy"), cliff_m.get("accuracy")),
    }
    _log.info(
        "cliff-aware: overall AUROC=%.3f | cliff-record AUROC=%.3f | non-cliff AUROC=%.3f | "
        "cliff-pair directional acc=%.3f",
        overall.get("auroc", float("nan")),
        cliff_m.get("auroc", float("nan")),
        non_m.get("auroc", float("nan")),
        pair_level.get("cliff_pair_directional_accuracy", float("nan")),
    )
    return {
        "overall": overall,
        "cliff_records": cliff_m,
        "non_cliff_records": non_m,
        "pair_level": pair_level,
        "gap": gap,
    }


def _safe_sub(a, b) -> float:
    if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float(a - b)
