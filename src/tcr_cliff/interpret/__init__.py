# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Model interpretation: SHAP field attribution, residue occlusion, anchors, plots.

Public surface
--------------
* :func:`explain_baseline` - SHAP over baseline embeddings, aggregated per field.
* :func:`residue_attribution` - model-agnostic occlusion importance per residue.
* :func:`attention_rollout` - optional torch attention path.
* :func:`anchor_importance` / :func:`map_to_positions` - peptide anchor analysis.
* :func:`plot_residue_importance` / :func:`plot_field_importance` /
  :func:`plot_cliff_graph` - headless PNG plots.
"""

from __future__ import annotations

from tcr_cliff.interpret.anchors import anchor_importance, map_to_positions
from tcr_cliff.interpret.attribution import (
    attention_rollout,
    residue_attribution,
)
from tcr_cliff.interpret.plots import (
    plot_cliff_graph,
    plot_field_importance,
    plot_residue_importance,
)
from tcr_cliff.interpret.shap_baseline import explain_baseline

__all__ = [
    "anchor_importance",
    "attention_rollout",
    "explain_baseline",
    "map_to_positions",
    "plot_cliff_graph",
    "plot_field_importance",
    "plot_residue_importance",
    "residue_attribution",
]
