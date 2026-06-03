# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Robust, literature-grounded cliff metrics.

A single random test split usually contains too few cliffs for stable per-record
AUROC (it collapses to NaN/0/1). These metrics fix that by (a) pooling *within-pair*
comparisons over all cliff pairs, (b) bootstrapping confidence intervals, (c)
weighting by cliff steepness, (d) measuring the model's prediction *responsiveness*
across cliffs vs. smooth neighbours, and (e) comparing against a similarity baseline.

References
----------
* van Tilborg, Alenicheva & Grisoni (2022), *J. Chem. Inf. Model.* 62(23):5938 -
  "Exposing the limitations of molecular machine learning with activity cliffs."
  Establishes cliff-specific evaluation and similarity baselines.
* Guha & Van Drie (2008), *J. Chem. Inf. Model.* 48(3):646 - SALI (Structure-Activity
  Landscape Index), the steepness measure we adapt to TCR sequence edit distance.
* Stumpfe & Bajorath, activity-cliff / matched-molecular-pair analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tcr_cliff._logging import get_logger
from tcr_cliff.cliffs.detect import NeighborPair
from tcr_cliff.cliffs.distance import bounded_levenshtein

_log = get_logger("eval.cliff_metrics")


@dataclass(frozen=True)
class EvaluatedPair:
    """A neighbour pair with the model's behaviour on it, ready for pooling.

    ``pos_score``/``neg_score`` are the model scores of the binder / non-binder of a
    label-flip pair (NaN for same-label pairs). ``delta_pred`` is ``|p_i - p_j|`` for
    *any* pair (used by the responsiveness gap). ``steepness`` is the SALI weight.
    """

    is_cliff: bool
    is_flip: bool
    pos_score: float
    neg_score: float
    delta_pred: float
    steepness: float


def pair_steepness(pair: NeighborPair, *, use_affinity: bool = False) -> float:
    """SALI-style steepness ``|Δactivity| / distance`` (Guha & Van Drie, 2008).

    For binary labels ``|Δactivity|`` is 0/1, so a single-residue label-flip cliff
    (distance 1) has steepness 1.0 and a 2-edit cliff 0.5 — the metric thus rewards
    getting the *sharpest* (single-residue) cliffs right. With affinities present and
    ``use_affinity=True`` the real ``|Δaffinity|`` is used.
    """
    if use_affinity and np.isfinite(pair.affinity_i) and np.isfinite(pair.affinity_j):
        delta = abs(pair.affinity_i - pair.affinity_j)
    else:
        delta = abs(pair.label_i - pair.label_j)
    return float(delta) / max(pair.distance, 1)


def evaluate_pairs(
    pairs: list[NeighborPair], y_score: np.ndarray, *, use_affinity: bool = False
) -> list[EvaluatedPair]:
    """Reduce ``(pairs, y_score)`` to poolable :class:`EvaluatedPair` records."""
    y_score = np.asarray(y_score, dtype=float)
    out: list[EvaluatedPair] = []
    for p in pairs:
        si, sj = float(y_score[p.i]), float(y_score[p.j])
        flip = p.label_i != p.label_j
        if flip:
            pos = si if p.label_i == 1 else sj
            neg = sj if p.label_i == 1 else si
        else:
            pos = neg = float("nan")
        out.append(
            EvaluatedPair(
                is_cliff=p.is_cliff,
                is_flip=flip,
                pos_score=pos,
                neg_score=neg,
                delta_pred=abs(si - sj),
                steepness=pair_steepness(p, use_affinity=use_affinity),
            )
        )
    return out


def cliff_auc(evaluated: list[EvaluatedPair], *, weighted: bool = False) -> float:
    """Pooled within-pair concordance over label-flip cliff pairs (a "Cliff-AUC").

    The probability the model scores the true binder above the non-binder across a
    cliff; ties count 0.5; 0.5 is chance. With ``weighted=True`` each pair is weighted
    by its SALI steepness so the steepest cliffs dominate.
    """
    num = den = 0.0
    for e in evaluated:
        if not (e.is_cliff and e.is_flip):
            continue
        w = e.steepness if weighted else 1.0
        if w <= 0:
            continue
        conc = 1.0 if e.pos_score > e.neg_score else (0.5 if e.pos_score == e.neg_score else 0.0)
        num += w * conc
        den += w
    return num / den if den else float("nan")


def cliff_responsiveness_gap(evaluated: list[EvaluatedPair]) -> dict[str, float]:
    """Cliff Responsiveness Gap (CRG): does the model move *more* across cliffs?

    ``CRG = mean(|Δp| | cliff) - mean(|Δp| | smooth)``. A cliff-aware model amplifies
    its prediction across opposite-outcome neighbours while staying smooth across
    same-outcome ones, so CRG > 0; a similarity-smooth model gives near-identical
    predictions to neighbours, so CRG ≈ 0. The normalised ``crg_effect_size`` is a
    Cohen's-d-style standardisation by the pooled standard deviation of ``|Δp|``.

    Uses *all* neighbour pairs (cliff and smooth), so it is far more data-efficient and
    stable than the label-flip-only metrics.
    """
    cliff_d = np.array([e.delta_pred for e in evaluated if e.is_cliff], dtype=float)
    smooth_d = np.array([e.delta_pred for e in evaluated if not e.is_cliff], dtype=float)
    if len(cliff_d) == 0 or len(smooth_d) == 0:
        return {
            "crg": float("nan"),
            "crg_effect_size": float("nan"),
            "mean_cliff_delta": float(cliff_d.mean()) if len(cliff_d) else float("nan"),
            "mean_smooth_delta": float(smooth_d.mean()) if len(smooth_d) else float("nan"),
            "n_cliff": float(len(cliff_d)),
            "n_smooth": float(len(smooth_d)),
        }
    gap = float(cliff_d.mean() - smooth_d.mean())
    pooled = np.concatenate([cliff_d, smooth_d])
    sd = float(pooled.std(ddof=1)) if len(pooled) > 1 else 0.0
    return {
        "crg": gap,
        "crg_effect_size": gap / sd if sd > 0 else float("nan"),
        "mean_cliff_delta": float(cliff_d.mean()),
        "mean_smooth_delta": float(smooth_d.mean()),
        "n_cliff": float(len(cliff_d)),
        "n_smooth": float(len(smooth_d)),
    }


def bootstrap_cliff_auc(
    evaluated: list[EvaluatedPair],
    *,
    weighted: bool = False,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap the Cliff-AUC over cliff pairs; returns ``(point, lo, hi)``."""
    flips = [e for e in evaluated if e.is_cliff and e.is_flip]
    point = cliff_auc(evaluated, weighted=weighted)
    if len(flips) < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(flips)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = [flips[i] for i in idx]
        val = cliff_auc(sample, weighted=weighted)
        if np.isfinite(val):
            stats.append(val)
    if not stats:
        return point, float("nan"), float("nan")
    return (
        point,
        float(np.percentile(stats, 100 * alpha / 2)),
        float(np.percentile(stats, 100 * (1 - alpha / 2))),
    )


def nearest_neighbor_scores(
    train_df: pd.DataFrame,
    query_df: pd.DataFrame,
    *,
    max_train: int = 4000,
    seed: int = 0,
) -> np.ndarray:
    """1-NN similarity baseline: each query's score is the binder label of its most
    sequence-similar training record (combined CDR3b + peptide edit distance).

    This is the memorisation baseline of van Tilborg et al. (2022): it must fail on
    cliffs *by construction*, because a cliff member's nearest neighbour is often its
    cliff partner with the opposite label. The model's lift over this baseline on
    cliffs is the evidence it learned more than similarity. ``max_train`` caps the
    reference set for tractability (sampled deterministically).
    """
    tr = train_df
    if len(tr) > max_train:
        tr = tr.sample(max_train, random_state=seed)
    t_cdr3 = tr["cdr3b"].astype(str).tolist()
    t_pep = tr["peptide"].astype(str).tolist()
    t_lab = tr["binder"].to_numpy().astype(float)
    # Block by exact peptide to shrink the candidate set (huge speed-up on real data
    # where few epitopes recur), falling back to the global set when a peptide is unseen.
    by_pep: dict[str, list[int]] = {}
    for k, pep in enumerate(t_pep):
        by_pep.setdefault(pep, []).append(k)

    scores = np.empty(len(query_df), dtype=float)
    q_cdr3 = query_df["cdr3b"].astype(str).tolist()
    q_pep = query_df["peptide"].astype(str).tolist()
    cap = 60  # distance cap for the bounded edit distance
    for qi in range(len(query_df)):
        cand = by_pep.get(q_pep[qi])
        if cand:  # same epitope: nearest by CDR3b only (peptide distance 0)
            best_d, best = cap + 1, []
            for k in cand:
                d = bounded_levenshtein(q_cdr3[qi], t_cdr3[k], cap)
                if d < best_d:
                    best_d, best = d, [k]
                elif d == best_d:
                    best.append(k)
            scores[qi] = float(np.mean(t_lab[best]))
        else:  # unseen epitope: combined CDR3b + peptide distance over all train
            best_d, best = cap * 2 + 2, []
            for k in range(len(t_cdr3)):
                d = bounded_levenshtein(q_cdr3[qi], t_cdr3[k], cap) + bounded_levenshtein(
                    q_pep[qi], t_pep[k], cap
                )
                if d < best_d:
                    best_d, best = d, [k]
                elif d == best_d:
                    best.append(k)
            scores[qi] = float(np.mean(t_lab[best])) if best else 0.5
    return scores


def enhanced_cliff_metrics(
    pairs: list[NeighborPair],
    y_score: np.ndarray,
    *,
    use_affinity: bool = False,
    n_boot: int = 500,
    seed: int = 0,
) -> dict:
    """Compute the full enhanced cliff-metric block for one set of predictions.

    Returns Cliff-AUC (with bootstrap CI), the SALI-weighted Cliff-AUC, and the Cliff
    Responsiveness Gap. Stable even when a single split has few cliffs.
    """
    ev = evaluate_pairs(pairs, y_score, use_affinity=use_affinity)
    point, lo, hi = bootstrap_cliff_auc(ev, n_boot=n_boot, seed=seed)
    crg = cliff_responsiveness_gap(ev)
    n_flip = sum(1 for e in ev if e.is_cliff and e.is_flip)
    return {
        "cliff_auc": point,
        "cliff_auc_lo": lo,
        "cliff_auc_hi": hi,
        "sali_weighted_cliff_auc": cliff_auc(ev, weighted=True),
        "n_cliff_flip_pairs": float(n_flip),
        **crg,
    }
