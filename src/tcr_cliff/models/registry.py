# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Unified model entry points used by the CLI and evaluation.

Every model conforms to the :class:`Model` protocol — ``predict_proba(df) -> scores``
plus ``save(dir)`` / ``load(dir)`` — so the rest of the package treats baseline,
cliff-aware, and fusion models interchangeably. The torch-backed models are imported
lazily so the baseline path needs no torch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from tcr_cliff._logging import get_logger
from tcr_cliff.config import Config
from tcr_cliff.models.baseline import BaselineModel

_log = get_logger("models.registry")


@runtime_checkable
class Model(Protocol):
    """Structural type every tcr-cliff model satisfies."""

    kind: str

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray: ...

    def save(self, path: str | Path) -> Path: ...


def train_model(cfg: Config, df_train: pd.DataFrame, df_val: pd.DataFrame | None = None) -> Model:
    """Train the model selected by ``cfg.model.kind`` and return it.

    * ``baseline_lgbm`` - embeddings + gradient boosting (no torch needed).
    * ``cliff_aware``   - contrastive cliff-aware pretraining + supervised head (torch).
    * ``fusion``        - cliff-aware head fused with structural-confidence features (torch).
    """
    kind = cfg.model.kind
    _log.info("training model kind=%s on %d rows", kind, len(df_train))
    if kind == "baseline_lgbm":
        return BaselineModel(cfg).fit(df_train, df_val)
    if kind == "cliff_aware":
        from tcr_cliff.models.head import CliffAwareModel  # lazy: requires torch

        return CliffAwareModel(cfg).fit(df_train, df_val)
    if kind == "fusion":
        from tcr_cliff.models.fusion import FusionModel  # lazy: requires torch

        return FusionModel(cfg).fit(df_train, df_val)
    raise ValueError(f"unknown model kind {kind!r}")


def load_model(path: str | Path) -> Model:
    """Load a saved model, dispatching on the ``kind`` recorded in ``meta.json``."""
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    kind = meta["kind"]
    if kind == "baseline_lgbm":
        return BaselineModel.load(path)
    if kind == "cliff_aware":
        from tcr_cliff.models.head import CliffAwareModel

        return CliffAwareModel.load(path)
    if kind == "fusion":
        from tcr_cliff.models.fusion import FusionModel

        return FusionModel.load(path)
    raise ValueError(f"unknown saved model kind {kind!r}")


def predict_scores(model: Model, df: pd.DataFrame) -> np.ndarray:
    """Return positive-class scores for ``df`` (thin wrapper for symmetry)."""
    return np.asarray(model.predict_proba(df), dtype=float)
