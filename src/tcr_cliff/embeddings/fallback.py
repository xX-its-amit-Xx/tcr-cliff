# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Dependency-free fallback embedder (no torch / fair-esm required).

This backend lets the entire pipeline—tests, CLI, and cookbook—run offline on a
minimal install. It is *not* a protein language model: it combines signed k-mer
hashing with per-residue physicochemical scales. Crucially it is **sensitive to
single-residue substitutions** (so activity cliffs are not collapsed to identical
vectors) and **position-aware** (so anchor-position signal survives), which is
exactly what the cliff-aware machinery needs to demonstrate behaviour without GPUs.
"""

from __future__ import annotations

import numpy as np

from tcr_cliff.embeddings.base import Embedder
from tcr_cliff.utils.seed import stable_hash

# Compact physicochemical scales (normalised to ~[-1, 1]); index by amino acid.
# Kyte-Doolittle hydropathy, side-chain volume, net charge, polarity, aromaticity, isoelectric.
_SCALES: dict[str, tuple[float, ...]] = {
    "A": (0.62, -0.74, 0.0, -0.5, 0.0, -0.1),
    "C": (0.29, -0.30, 0.0, -0.2, 0.0, -0.3),
    "D": (-0.90, -0.36, -1.0, 1.0, 0.0, -1.0),
    "E": (-0.74, 0.08, -1.0, 0.9, 0.0, -0.9),
    "F": (1.00, 0.86, 0.0, -0.8, 1.0, -0.1),
    "G": (0.48, -1.00, 0.0, -0.3, 0.0, -0.1),
    "H": (-0.40, 0.32, 0.5, 0.4, 0.5, 0.3),
    "I": (1.38, 0.55, 0.0, -0.8, 0.0, -0.1),
    "K": (-1.50, 0.55, 1.0, 0.8, 0.0, 1.0),
    "L": (1.06, 0.55, 0.0, -0.8, 0.0, -0.1),
    "M": (0.64, 0.43, 0.0, -0.6, 0.0, -0.1),
    "N": (-0.78, -0.20, 0.0, 0.6, 0.0, -0.1),
    "P": (0.12, -0.39, 0.0, -0.3, 0.0, 0.0),
    "Q": (-0.85, 0.13, 0.0, 0.5, 0.0, -0.1),
    "R": (-2.53, 0.77, 1.0, 0.9, 0.0, 1.0),
    "S": (-0.18, -0.59, 0.0, 0.2, 0.0, -0.1),
    "T": (-0.05, -0.26, 0.0, 0.1, 0.0, -0.1),
    "V": (1.08, 0.20, 0.0, -0.7, 0.0, -0.1),
    "W": (0.81, 1.00, 0.0, -0.6, 1.0, -0.1),
    "Y": (0.26, 0.87, 0.0, -0.3, 1.0, -0.2),
}
_N_SCALES = 6
_AUX = 8  # reserved tail dims: physicochemical means(6) + length + nonempty flag


class FallbackEmbedder(Embedder):
    """Signed k-mer hashing + physicochemical features. Pure NumPy, deterministic."""

    def __init__(self, dim: int = 320, kmer_k: int = 3, seed: int = 0) -> None:
        if dim <= _AUX + 4:
            raise ValueError(f"dim must exceed {_AUX + 4}")
        self._dim = int(dim)
        self._k = int(kmer_k)
        self._hash_dim = self._dim - _AUX
        self.name = f"fallback-d{self._dim}-k{self._k}"
        self._seed = seed

    @property
    def dim(self) -> int:
        return self._dim

    def _bucket(self, token: str) -> tuple[int, float]:
        """Deterministically map a token to a (bucket, sign) pair."""
        h = int(stable_hash(f"{self._seed}:{token}", length=12), 16)
        return h % self._hash_dim, 1.0 if (h >> 8) & 1 else -1.0

    def _embed_mean(self, sequences: list[str]) -> np.ndarray:
        out = np.zeros((len(sequences), self._dim), dtype=np.float32)
        for n, seq in enumerate(sequences):
            seq = seq.upper()
            v = out[n]
            if not seq:
                continue
            # Signed k-mer hashing for k = 1..kmer_k. Smaller k -> more weight (robust to noise).
            for k in range(1, self._k + 1):
                w = 1.0 / k
                for i in range(len(seq) - k + 1):
                    idx, sign = self._bucket(seq[i : i + k])
                    v[idx] += sign * w
            # Physicochemical means over the sequence (the AUX tail).
            scales = np.array([_SCALES.get(c, (0.0,) * _N_SCALES) for c in seq], dtype=np.float32)
            v[self._hash_dim : self._hash_dim + _N_SCALES] = scales.mean(axis=0)
            v[self._hash_dim + _N_SCALES] = len(seq) / 32.0
            v[self._hash_dim + _N_SCALES + 1] = 1.0
            # L2 normalise the hashing block for scale stability.
            block = v[: self._hash_dim]
            norm = np.linalg.norm(block)
            if norm > 0:
                v[: self._hash_dim] = block / norm
        return out

    def _embed_per_residue(self, sequences: list[str]) -> list[np.ndarray]:
        d_res = _N_SCALES + 2 + 16  # physchem + (pos, len) + hashed identity/context
        results: list[np.ndarray] = []
        for seq in sequences:
            seq = seq.upper()
            if not seq:
                results.append(np.zeros((0, d_res), dtype=np.float32))
                continue
            mat = np.zeros((len(seq), d_res), dtype=np.float32)
            L = len(seq)
            for i, c in enumerate(seq):
                mat[i, :_N_SCALES] = _SCALES.get(c, (0.0,) * _N_SCALES)
                mat[i, _N_SCALES] = i / max(L - 1, 1)  # normalised position
                mat[i, _N_SCALES + 1] = L / 32.0
                # local 3-mer context hashed into the remaining 16 dims
                ctx = seq[max(0, i - 1) : i + 2]
                idx = int(stable_hash(ctx, length=8), 16) % 16
                mat[i, _N_SCALES + 2 + idx] += 1.0
            results.append(mat)
        return results
