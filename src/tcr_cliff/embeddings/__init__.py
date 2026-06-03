# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Sequence embeddings for CDR3-beta, peptide, and (optional) MHC pseudo-sequence.

The public entry points are :func:`get_embedder`, :func:`embed_pairs`, and
:func:`build_feature_matrix`. The ESM-2 backends are imported lazily so that the
fallback path has no heavy dependencies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tcr_cliff._logging import get_logger
from tcr_cliff.config import EmbeddingConfig
from tcr_cliff.embeddings.base import Embedder, EmbeddingResult
from tcr_cliff.embeddings.cache import cached_embed
from tcr_cliff.embeddings.fallback import FallbackEmbedder

_log = get_logger("embeddings")

__all__ = [
    "Embedder",
    "EmbeddingResult",
    "FallbackEmbedder",
    "build_feature_matrix",
    "cached_embed",
    "embed_pairs",
    "get_embedder",
]


def get_embedder(cfg: EmbeddingConfig) -> Embedder:
    """Construct the embedder selected by ``cfg.backend``.

    * ``fallback`` - pure-NumPy embedder (no extra deps).
    * ``esm-fair`` - ESM-2 via the ``fair-esm`` package.
    * ``esm-hf``   - ESM-2 via HuggingFace ``transformers``.

    The ESM backends are imported lazily and raise a clear, actionable error if
    their optional dependencies are not installed.
    """
    if cfg.backend == "fallback":
        return FallbackEmbedder(dim=cfg.fallback_dim, kmer_k=cfg.kmer_k)
    if cfg.backend in ("esm-fair", "esm-hf"):
        from tcr_cliff.embeddings.esm import ESMEmbedder  # lazy: optional deps

        return ESMEmbedder(
            model_name=cfg.model_name,
            backend=cfg.backend,
            device=cfg.device,
            layers=cfg.layers,
            batch_size=cfg.batch_size,
            max_len=cfg.max_len,
        )
    raise ValueError(f"unknown embedding backend {cfg.backend!r}")


def embed_pairs(
    df: pd.DataFrame,
    cfg: EmbeddingConfig,
    *,
    fields: tuple[str, ...] | None = None,
    pooling: str | None = None,
) -> dict[str, EmbeddingResult]:
    """Embed each configured sequence field of a pairs table.

    Returns a mapping ``field -> EmbeddingResult``. Sequences are de-duplicated via
    the on-disk cache so overlapping fields/rows are not recomputed.
    """
    fields = fields or cfg.fields
    pooling = pooling or cfg.pooling
    embedder = get_embedder(cfg)
    out: dict[str, EmbeddingResult] = {}
    for fld in fields:
        if fld not in df.columns:
            _log.warning("field %s missing from table; skipping", fld)
            continue
        seqs = df[fld].fillna("").astype(str).tolist()
        out[fld] = cached_embed(embedder, seqs, pooling=pooling, cache_dir=cfg.cache_dir)
        _log.info(
            "embedded %d %s sequences -> dim %d (%s)", len(seqs), fld, out[fld].dim, embedder.name
        )
    return out


def build_feature_matrix(
    df: pd.DataFrame, cfg: EmbeddingConfig, *, fields: tuple[str, ...] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Concatenate mean-pooled per-field embeddings into one feature matrix.

    Returns
    -------
    (X, names):
        ``X`` is ``[N, sum_f D_f]``; ``names`` are ``"<field>_<j>"`` column labels,
        used downstream for SHAP attribution back onto each sequence source.
    """
    fields = fields or cfg.fields
    embs = embed_pairs(df, cfg, fields=fields, pooling="mean")
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for fld in fields:
        if fld not in embs:
            continue
        mat = embs[fld].vectors
        blocks.append(mat)
        names.extend(f"{fld}_{j}" for j in range(mat.shape[1]))
    if not blocks:
        raise ValueError("no embeddable fields produced features")
    return np.hstack(blocks).astype(np.float32), names
