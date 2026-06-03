# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Summary statistics over detected cliffs."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from tcr_cliff.cliffs.detect import NeighborPair


def cliff_statistics(pairs: list[NeighborPair], n_records: int | None = None) -> dict:
    """Compute aggregate statistics over a list of neighbour pairs.

    Returns a JSON-serialisable dict with counts, the cliff ratio (a dataset's
    "ruggedness"), distance/vary/reason breakdowns, and the set of records that
    participate in at least one cliff.
    """
    n_pairs = len(pairs)
    cliffs = [p for p in pairs if p.is_cliff]
    n_cliff = len(cliffs)
    cliff_nodes: set = set()
    for p in cliffs:
        cliff_nodes.update((p.id_i, p.id_j))

    by_vary = Counter(p.vary for p in cliffs)
    by_dist = Counter(p.distance for p in cliffs)
    by_reason = Counter(p.reason for p in cliffs)
    aff_deltas = [
        abs(p.affinity_i - p.affinity_j)
        for p in cliffs
        if np.isfinite(p.affinity_i) and np.isfinite(p.affinity_j)
    ]

    return {
        "n_neighbor_pairs": n_pairs,
        "n_cliff_pairs": n_cliff,
        "n_smooth_pairs": n_pairs - n_cliff,
        "cliff_ratio": (n_cliff / n_pairs) if n_pairs else 0.0,
        "n_records": n_records,
        "n_records_in_cliff": len(cliff_nodes),
        "frac_records_in_cliff": (len(cliff_nodes) / n_records) if n_records else None,
        "cliffs_by_vary": dict(by_vary),
        "cliffs_by_distance": {int(k): int(v) for k, v in by_dist.items()},
        "cliffs_by_reason": dict(by_reason),
        "affinity_delta_mean": float(np.mean(aff_deltas)) if aff_deltas else None,
        "affinity_delta_max": float(np.max(aff_deltas)) if aff_deltas else None,
    }


def per_record_cliff_density(pairs: list[NeighborPair], df: pd.DataFrame) -> pd.DataFrame:
    """Per-record counts of incident cliff and smooth edges."""
    ids = df["pair_id"].tolist() if "pair_id" in df else [str(i) for i in range(len(df))]
    cliff_deg = dict.fromkeys(ids, 0)
    smooth_deg = dict.fromkeys(ids, 0)
    for p in pairs:
        tgt = cliff_deg if p.is_cliff else smooth_deg
        tgt[p.id_i] = tgt.get(p.id_i, 0) + 1
        tgt[p.id_j] = tgt.get(p.id_j, 0) + 1
    return pd.DataFrame(
        {
            "pair_id": ids,
            "cliff_degree": [cliff_deg[i] for i in ids],
            "smooth_degree": [smooth_deg[i] for i in ids],
        }
    )
