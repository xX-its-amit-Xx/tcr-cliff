# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Binding-cliff definition, detection, graphs, and statistics."""

from __future__ import annotations

from tcr_cliff.cliffs.detect import (
    NeighborPair,
    cliff_pairs_to_frame,
    find_cliffs,
    find_neighbor_pairs,
    mark_cliff_membership,
)
from tcr_cliff.cliffs.distance import bounded_levenshtein, hamming, levenshtein, within_k
from tcr_cliff.cliffs.graph import (
    build_cliff_graph,
    cliff_components,
    cliff_hubs,
    export_graph,
)
from tcr_cliff.cliffs.stats import cliff_statistics, per_record_cliff_density

__all__ = [
    "NeighborPair",
    "bounded_levenshtein",
    "build_cliff_graph",
    "cliff_components",
    "cliff_hubs",
    "cliff_pairs_to_frame",
    "cliff_statistics",
    "export_graph",
    "find_cliffs",
    "find_neighbor_pairs",
    "hamming",
    "levenshtein",
    "mark_cliff_membership",
    "per_record_cliff_density",
    "within_k",
]
