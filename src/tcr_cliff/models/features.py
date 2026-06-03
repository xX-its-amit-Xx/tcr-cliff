# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Feature assembly: sequence embeddings (+ optional structure-confidence features).

This is the single place models obtain their inputs, so the baseline, the
cliff-aware model, and the fusion model all see consistent, named feature blocks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tcr_cliff._logging import get_logger
from tcr_cliff.config import Config
from tcr_cliff.embeddings import build_feature_matrix

_log = get_logger("models.features")


def sequence_features(df: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, list[str]]:
    """Mean-pooled per-field sequence embeddings concatenated into one matrix."""
    return build_feature_matrix(df, cfg.embedding)


def structure_features(df: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, list[str]]:
    """Structural-confidence features per pair (zeros if structure is disabled).

    Imported lazily so the structure module's dependencies are only needed when
    fusion/structure is actually requested.
    """
    from tcr_cliff.structure import load_structure_features  # lazy

    return load_structure_features(df, cfg.structure)


def assemble_features(
    df: pd.DataFrame, cfg: Config, *, with_structure: bool = False
) -> tuple[np.ndarray, list[str]]:
    """Concatenate sequence features and (optionally) structure features.

    Returns ``(X, names)``. ``with_structure=True`` is used by the fusion model.
    """
    X_seq, names = sequence_features(df, cfg)
    if not with_structure:
        return X_seq, names
    X_struct, struct_names = structure_features(df, cfg)
    X = np.hstack([X_seq, X_struct]).astype(np.float32)
    return X, [*names, *struct_names]
