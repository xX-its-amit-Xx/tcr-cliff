# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Models: baseline (boosting) + cliff-aware contrastive + structure-fusion.

Only the baseline and the shared registry/feature code are imported eagerly; the
torch-backed models are pulled in lazily by :func:`train_model` / :func:`load_model`.
"""

from __future__ import annotations

from tcr_cliff.models.baseline import BaselineModel, lightgbm_available
from tcr_cliff.models.features import (
    assemble_features,
    sequence_features,
    structure_features,
)
from tcr_cliff.models.registry import Model, load_model, predict_scores, train_model

__all__ = [
    "BaselineModel",
    "Model",
    "assemble_features",
    "lightgbm_available",
    "load_model",
    "predict_scores",
    "sequence_features",
    "structure_features",
    "train_model",
]
