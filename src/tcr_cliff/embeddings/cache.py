# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Disk caching for embeddings, keyed by (backend, pooling, sequence)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tcr_cliff._logging import get_logger
from tcr_cliff.embeddings.base import Embedder, EmbeddingResult
from tcr_cliff.utils.seed import stable_hash

_log = get_logger("embeddings.cache")


def _key(backend: str, pooling: str, seq: str) -> str:
    return stable_hash(f"{backend}|{pooling}|{seq}", length=24)


def cached_embed(
    embedder: Embedder,
    sequences: list[str],
    *,
    pooling: str = "mean",
    cache_dir: str | Path | None = None,
) -> EmbeddingResult:
    """Embed ``sequences``, reusing per-sequence results cached on disk.

    Each unique sequence is stored as a single ``.npy`` file so repeated runs over
    overlapping datasets (the common case across embed/train/eval) only compute the
    novel sequences. With ``cache_dir=None`` caching is disabled.
    """
    if cache_dir is None:
        return embedder.embed(sequences, pooling=pooling)

    cache_dir = Path(cache_dir) / embedder.name / pooling
    cache_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, np.ndarray] = {}
    missing_idx: list[int] = []
    missing_seq: list[str] = []
    for i, seq in enumerate(sequences):
        fp = cache_dir / f"{_key(embedder.name, pooling, seq)}.npy"
        if fp.exists():
            results[i] = np.load(fp)
        else:
            missing_idx.append(i)
            missing_seq.append(seq)

    if missing_seq:
        _log.debug("computing %d/%d uncached embeddings", len(missing_seq), len(sequences))
        fresh = embedder.embed(missing_seq, pooling=pooling)
        for j, i in enumerate(missing_idx):
            arr = fresh.vectors[j] if pooling == "mean" else fresh.per_residue[j]
            results[i] = arr
            fp = cache_dir / f"{_key(embedder.name, pooling, sequences[i])}.npy"
            np.save(fp, arr)

    ordered = [results[i] for i in range(len(sequences))]
    if pooling == "mean":
        vecs = (
            np.vstack(ordered).astype(np.float32)
            if ordered
            else np.empty((0, embedder.dim), np.float32)
        )
        return EmbeddingResult(list(sequences), embedder.name, vecs.shape[1], vectors=vecs)
    d = ordered[0].shape[1] if ordered else embedder.dim
    return EmbeddingResult(list(sequences), embedder.name, d, per_residue=ordered)
