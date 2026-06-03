# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Embedding abstractions shared by every backend.

All embedders return :class:`EmbeddingResult` objects so downstream code is
backend-agnostic. ``mean`` pooling yields one vector per sequence; ``per_residue``
pooling yields a list of ``[L_i, d]`` arrays (used by the attribution code that
maps importance back onto CDR3/peptide positions).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EmbeddingResult:
    """Container for an embedding batch.

    Attributes
    ----------
    vectors:
        ``[N, D]`` matrix when ``pooling == 'mean'``; otherwise an empty array.
    per_residue:
        List of ``[L_i, d]`` arrays when ``pooling == 'per_residue'``; else empty.
    sequences:
        The input sequences, in order.
    dim:
        Embedding dimensionality ``D`` (mean) or ``d`` (per-residue).
    backend:
        Identifier of the backend that produced the embeddings.
    """

    sequences: list[str]
    backend: str
    dim: int
    vectors: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))
    per_residue: list[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.sequences)


class Embedder(abc.ABC):
    """Abstract sequence embedder.

    Concrete subclasses implement :meth:`_embed_mean` and (optionally)
    :meth:`_embed_per_residue`. ``embed`` dispatches on the requested pooling.
    """

    #: Short, stable identifier used in cache keys (e.g. ``"fallback-d320-k3"``).
    name: str = "embedder"

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Mean-pooled embedding dimensionality."""

    @abc.abstractmethod
    def _embed_mean(self, sequences: list[str]) -> np.ndarray:
        """Return a ``[N, D]`` float32 matrix of mean-pooled embeddings."""

    def _embed_per_residue(self, sequences: list[str]) -> list[np.ndarray]:
        """Return per-residue ``[L_i, d]`` arrays. Default: not supported."""
        raise NotImplementedError(f"{self.name} does not support per-residue pooling")

    def embed(self, sequences: list[str], *, pooling: str = "mean") -> EmbeddingResult:
        """Embed a list of sequences with the requested pooling."""
        seqs = [s if isinstance(s, str) else "" for s in sequences]
        if pooling == "mean":
            vecs = self._embed_mean(seqs).astype(np.float32, copy=False)
            return EmbeddingResult(
                seqs, self.name, vecs.shape[1] if vecs.size else self.dim, vectors=vecs
            )
        if pooling == "per_residue":
            pr = self._embed_per_residue(seqs)
            d = pr[0].shape[1] if pr else self.dim
            return EmbeddingResult(seqs, self.name, d, per_residue=pr)
        raise ValueError(f"unknown pooling {pooling!r}")
