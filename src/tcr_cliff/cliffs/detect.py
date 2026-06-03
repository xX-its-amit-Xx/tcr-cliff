# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Detect TCR-pMHC binding cliffs and their smooth-neighbour controls.

A **binding cliff** is a pair of records whose varied sequence (peptide or CDR3-beta)
is within ``k`` edits, but whose binding outcome differs — a label flip and/or an
affinity jump above a threshold — while the *context* (the non-varied entity and the
MHC) is held fixed. This is the altered-peptide-ligand / single-residue-cross-reactivity
phenomenon that the package is built around.

The same machinery also surfaces **smooth neighbours** (within ``k`` edits, *same*
outcome). Cliffs vs. smooth neighbours are exactly the positive/negative pairs the
contrastive cliff-aware model trains on, and the strata the cliff-aware evaluation
compares.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tcr_cliff._logging import get_logger
from tcr_cliff.cliffs.distance import bounded_levenshtein, within_k
from tcr_cliff.config import CliffConfig

_log = get_logger("cliffs.detect")

_CONTEXT_FOR = {"peptide": ("cdr3b", "mhc"), "cdr3b": ("peptide", "mhc")}


@dataclass(frozen=True)
class NeighborPair:
    """A within-``k`` pair of records and whether it is a cliff."""

    i: int
    j: int
    id_i: str
    id_j: str
    vary: str  # 'peptide' or 'cdr3b'
    seq_i: str
    seq_j: str
    distance: int
    label_i: int
    label_j: int
    affinity_i: float
    affinity_j: float
    is_cliff: bool
    reason: str  # 'label_flip' | 'affinity_delta' | 'label_flip+affinity' | 'smooth'

    @property
    def key(self) -> tuple[int, int, str]:
        lo, hi = sorted((self.i, self.j))
        return (lo, hi, self.vary)


def _cliff_decision(
    cfg: CliffConfig, label_i: int, label_j: int, aff_i: float, aff_j: float
) -> tuple[bool, str]:
    """Apply the cliff definition; return ``(is_cliff, reason)``."""
    flip = label_i != label_j
    aff_ok = (
        cfg.affinity_delta is not None
        and np.isfinite(aff_i)
        and np.isfinite(aff_j)
        and abs(aff_i - aff_j) > cfg.affinity_delta
    )
    conditions = []
    if cfg.require_label_flip:
        conditions.append(("label_flip", flip))
    if cfg.affinity_delta is not None:
        conditions.append(("affinity_delta", aff_ok))
    if not conditions:  # neither criterion configured -> default to label flip
        conditions.append(("label_flip", flip))

    reasons = [name for name, ok in conditions if ok]
    is_cliff = len(reasons) > 0
    if not is_cliff:
        return False, "smooth"
    return True, "+".join(reasons)


def _candidate_pairs_k1_levenshtein(seqs: list[str]) -> Iterable[tuple[int, int]]:
    """Yield candidate local index pairs within Levenshtein-1 via a deletion index.

    Two strings are within edit distance 1 iff they share a common "deletion
    variant" (the string with at most one character removed) — the SymSpell trick.
    This avoids the O(m^2) all-pairs scan when a context group is large.
    """
    index: dict[str, list[int]] = defaultdict(list)
    for idx, s in enumerate(seqs):
        index[s].append(idx)  # identity variant (distance 0 / substitutions of equal length)
        for d in range(len(s)):
            index[s[:d] + s[d + 1 :]].append(idx)
    seen: set[tuple[int, int]] = set()
    for members in index.values():
        if len(members) < 2:
            continue
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                lo, hi = sorted((members[a], members[b]))
                if lo != hi:
                    seen.add((lo, hi))
    return seen


def _pairs_within_k(seqs: list[str], k: int, metric: str) -> list[tuple[int, int, int]]:
    """Return ``(i, j, distance)`` for all local pairs within ``k`` edits."""
    n = len(seqs)
    out: list[tuple[int, int, int]] = []
    # Fast path: k==1 Levenshtein on a large group uses the deletion index.
    if metric == "levenshtein" and k == 1 and n > 64:
        for i, j in _candidate_pairs_k1_levenshtein(seqs):
            d = bounded_levenshtein(seqs[i], seqs[j], k)
            if d <= k:
                out.append((i, j, d))
        return out
    # General path: bounded pairwise comparison with a length prefilter.
    for i in range(n):
        si = seqs[i]
        for j in range(i + 1, n):
            sj = seqs[j]
            if abs(len(si) - len(sj)) > k:
                continue
            ok, d = within_k(si, sj, k, metric)
            if ok:
                out.append((i, j, d))
    return out


def find_neighbor_pairs(df: pd.DataFrame, cfg: CliffConfig | None = None) -> list[NeighborPair]:
    """Find all within-``k`` neighbour pairs (cliffs and smooth) in a pairs table.

    Pairs are searched separately for each varied entity selected by ``cfg.vary``.
    When ``cfg.same_context`` is True (default), candidates are grouped by the
    non-varied entity and MHC so that only genuine single-axis neighbours are
    compared — this is both correct and fast.
    """
    cfg = cfg or CliffConfig()
    vary_entities = ("peptide", "cdr3b") if cfg.vary == "both" else (cfg.vary,)
    ids = df["pair_id"].tolist() if "pair_id" in df else [str(i) for i in range(len(df))]
    labels = df["binder"].to_numpy()
    aff = df["affinity"].to_numpy() if "affinity" in df else np.full(len(df), np.nan)

    results: dict[tuple[int, int, str], NeighborPair] = {}
    for vary in vary_entities:
        seq_all = df[vary].fillna("").astype(str).tolist()
        ctx_cols = [c for c in _CONTEXT_FOR[vary] if c in df.columns] if cfg.same_context else []
        # Build context groups (global indices).
        groups: dict[tuple, list[int]] = defaultdict(list)
        if ctx_cols:
            ctx_vals = list(zip(*[df[c].fillna("").astype(str).tolist() for c in ctx_cols]))
            for gi, key in enumerate(ctx_vals):
                if len(seq_all[gi]) >= cfg.min_len:
                    groups[key].append(gi)
        else:
            groups[("__all__",)] = [gi for gi in range(len(df)) if len(seq_all[gi]) >= cfg.min_len]

        for members in groups.values():
            if len(members) < 2:
                continue
            local_seqs = [seq_all[g] for g in members]
            for li, lj, dist in _pairs_within_k(local_seqs, cfg.max_edits, cfg.distance):
                gi, gj = members[li], members[lj]
                if seq_all[gi] == seq_all[gj]:
                    continue  # identical varied sequence is not a cliff
                is_cliff, reason = _cliff_decision(
                    cfg, int(labels[gi]), int(labels[gj]), float(aff[gi]), float(aff[gj])
                )
                pair = NeighborPair(
                    i=gi,
                    j=gj,
                    id_i=ids[gi],
                    id_j=ids[gj],
                    vary=vary,
                    seq_i=seq_all[gi],
                    seq_j=seq_all[gj],
                    distance=int(dist),
                    label_i=int(labels[gi]),
                    label_j=int(labels[gj]),
                    affinity_i=float(aff[gi]),
                    affinity_j=float(aff[gj]),
                    is_cliff=is_cliff,
                    reason=reason,
                )
                results[pair.key] = pair

    pairs = list(results.values())
    n_cliff = sum(p.is_cliff for p in pairs)
    _log.info(
        "found %d neighbour pairs (%d cliffs / %d smooth) over vary=%s",
        len(pairs),
        n_cliff,
        len(pairs) - n_cliff,
        cfg.vary,
    )
    return pairs


def find_cliffs(df: pd.DataFrame, cfg: CliffConfig | None = None) -> list[NeighborPair]:
    """Return only the cliff pairs (convenience wrapper over neighbour search)."""
    return [p for p in find_neighbor_pairs(df, cfg) if p.is_cliff]


def cliff_pairs_to_frame(pairs: list[NeighborPair]) -> pd.DataFrame:
    """Render neighbour/cliff pairs as a tidy DataFrame for export/inspection."""
    return pd.DataFrame([p.__dict__ for p in pairs])


def mark_cliff_membership(df: pd.DataFrame, pairs: list[NeighborPair]) -> pd.DataFrame:
    """Annotate each record with whether it participates in any cliff.

    Adds boolean columns ``in_cliff`` and ``in_neighbor`` keyed by ``pair_id`` /
    row index, used by the cliff-aware evaluation to stratify the test set.
    """
    df = df.copy()
    in_cliff = np.zeros(len(df), dtype=bool)
    in_neighbor = np.zeros(len(df), dtype=bool)
    for p in pairs:
        in_neighbor[p.i] = in_neighbor[p.j] = True
        if p.is_cliff:
            in_cliff[p.i] = in_cliff[p.j] = True
    df["in_cliff"] = in_cliff
    df["in_neighbor"] = in_neighbor
    return df
