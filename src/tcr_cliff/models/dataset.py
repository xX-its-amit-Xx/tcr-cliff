# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Datasets and batching helpers for the torch-backed cliff-aware models.

This module bridges the cliff-mining machinery (:func:`find_neighbor_pairs`) and
the sequence-feature assembly (:func:`sequence_features`) into the tensors the
contrastive pretraining and supervised head consume. ``torch`` is imported
lazily inside functions/methods so importing this module never requires torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from tcr_cliff._logging import get_logger
from tcr_cliff.cliffs.detect import NeighborPair, find_neighbor_pairs
from tcr_cliff.config import Config
from tcr_cliff.models.features import sequence_features

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import torch

_log = get_logger("models.dataset")


def build_pair_index(df: pd.DataFrame, cfg: Config) -> list[NeighborPair]:
    """Mine within-``k`` neighbour pairs (cliffs + smooth) from ``df``.

    Thin wrapper over :func:`find_neighbor_pairs` using ``cfg.cliff`` so the model
    code has a single, named entry point. The returned pairs carry positional row
    indices (``i``/``j``) into ``df`` and an ``is_cliff`` flag, which is exactly
    what the contrastive objective needs.

    Parameters
    ----------
    df:
        A normalised pairs table.
    cfg:
        Run configuration; ``cfg.cliff`` selects the neighbour/cliff definition.

    Returns
    -------
    list of NeighborPair
        All discovered neighbour pairs.
    """
    pairs = find_neighbor_pairs(df, cfg.cliff)
    n_cliff = sum(p.is_cliff for p in pairs)
    _log.info(
        "built pair index: %d pairs (%d cliff / %d smooth)",
        len(pairs),
        n_cliff,
        len(pairs) - n_cliff,
    )
    return pairs


def _build_contrastive_pairs_class() -> type:
    """Build the ``ContrastivePairs`` class bound to ``torch.utils.data.Dataset``."""
    torch = _import_torch()

    class _ContrastivePairs(torch.utils.data.Dataset):
        """A ``Dataset`` over neighbour pairs for contrastive training.

        Each item is ``(x_i, x_j, is_cliff_float)`` where ``x_i``/``x_j`` are the
        pre-computed sequence-feature rows for the two records of a neighbour pair
        and ``is_cliff_float`` is ``1.0`` for cliff pairs (push apart) and ``0.0``
        for smooth pairs (pull together).

        Parameters
        ----------
        X:
            Feature matrix ``[N, D]`` aligned to the rows of the source DataFrame.
        pairs:
            Neighbour pairs whose ``i``/``j`` index into ``X``.
        """

        def __init__(self, X: np.ndarray, pairs: list[NeighborPair]) -> None:
            self._x = torch.as_tensor(np.asarray(X, dtype=np.float32))
            self._pairs = list(pairs)
            self.dim = int(self._x.shape[1]) if self._x.ndim == 2 else 0

        def __len__(self) -> int:
            return len(self._pairs)

        def __getitem__(self, idx: int) -> Any:
            pair = self._pairs[idx]
            x_i = self._x[pair.i]
            x_j = self._x[pair.j]
            is_cliff = torch.tensor(1.0 if pair.is_cliff else 0.0, dtype=torch.float32)
            return x_i, x_j, is_cliff

    return _ContrastivePairs


_PAIRS_CLS: type | None = None


def ContrastivePairs(X: np.ndarray, pairs: list[NeighborPair]) -> Any:
    """Instantiate the contrastive-pairs dataset (lazily binds to torch).

    Parameters
    ----------
    X:
        Feature matrix ``[N, D]`` aligned to the source DataFrame rows.
    pairs:
        Neighbour pairs whose ``i``/``j`` index into ``X``; ``is_cliff`` selects
        whether each pair is pushed apart (cliff) or pulled together (smooth).

    Returns
    -------
    torch.utils.data.Dataset
        A dataset yielding ``(x_i, x_j, is_cliff_float)`` items.
    """
    global _PAIRS_CLS
    if _PAIRS_CLS is None:
        _PAIRS_CLS = _build_contrastive_pairs_class()
    return _PAIRS_CLS(X, pairs)


def make_pair_loader(
    dataset: ContrastivePairs,
    batch_size: int = 64,
    *,
    shuffle: bool = True,
) -> Any:
    """Build a ``DataLoader`` over a :class:`ContrastivePairs` dataset.

    Parameters
    ----------
    dataset:
        The contrastive-pair dataset.
    batch_size:
        Pairs per batch.
    shuffle:
        Whether to shuffle pairs each epoch.

    Returns
    -------
    torch.utils.data.DataLoader
        Yields ``(x_i[B, D], x_j[B, D], is_cliff[B])`` batches.
    """
    torch = _import_torch()
    bs = max(1, min(batch_size, max(1, len(dataset))))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle and len(dataset) > 1,
        drop_last=False,
    )


def labels_tensor(df: pd.DataFrame) -> Any:
    """Return the ``binder`` column as a float ``[N]`` tensor for BCE training."""
    torch = _import_torch()
    y = df["binder"].to_numpy().astype(np.float32)
    return torch.as_tensor(y)


def features_tensor(df: pd.DataFrame, cfg: Config) -> tuple[torch.Tensor, list[str]]:
    """Compute sequence features for ``df`` and return ``(X_tensor, names)``.

    Parameters
    ----------
    df:
        A normalised pairs table.
    cfg:
        Run configuration (``cfg.embedding`` drives feature assembly).

    Returns
    -------
    (torch.Tensor, list of str)
        Float feature tensor ``[N, D]`` and the per-column feature names.
    """
    torch = _import_torch()
    X, names = sequence_features(df, cfg)
    return torch.as_tensor(np.asarray(X, dtype=np.float32)), names


def _import_torch() -> Any:
    """Import torch lazily, raising an actionable error if it is missing."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "PyTorch is required for the cliff-aware/fusion models. "
            "Install it with: pip install 'tcr-cliff[torch]' (or `pip install torch`)."
        ) from exc
    return torch
