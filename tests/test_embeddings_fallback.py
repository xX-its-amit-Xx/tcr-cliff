# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Tests for the dependency-free fallback embedder and caching."""

from __future__ import annotations

import numpy as np

from tcr_cliff.config import EmbeddingConfig
from tcr_cliff.embeddings import build_feature_matrix, get_embedder
from tcr_cliff.embeddings.cache import cached_embed
from tcr_cliff.embeddings.fallback import FallbackEmbedder


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def test_fallback_deterministic():
    e = FallbackEmbedder(dim=128, kmer_k=3)
    v1 = e.embed(["GILGFVFTL"]).vectors
    v2 = e.embed(["GILGFVFTL"]).vectors
    assert np.allclose(v1, v2)
    assert v1.shape == (1, 128)


def test_single_residue_change_is_distinct_but_close():
    e = FallbackEmbedder(dim=256, kmer_k=3)
    base = e.embed(["GILGFVFTL"]).vectors[0]
    neighbor = e.embed(["GILGFVFTA"]).vectors[0]  # 1 substitution
    distant = e.embed(["QWERTYIPK"]).vectors[0]  # unrelated
    # The cliff partner must NOT collapse to an identical vector...
    assert not np.allclose(base, neighbor)
    # ...yet be closer than an unrelated sequence.
    assert _cos(base, neighbor) > _cos(base, distant)


def test_per_residue_shapes():
    e = FallbackEmbedder(dim=128)
    res = e.embed(["GILGFVFTL", "CASSF"], pooling="per_residue")
    assert len(res.per_residue) == 2
    assert res.per_residue[0].shape[0] == 9
    assert res.per_residue[1].shape[0] == 5
    assert res.per_residue[0].shape[1] == res.per_residue[1].shape[1]


def test_build_feature_matrix(toy_df):
    cfg = EmbeddingConfig(backend="fallback", fallback_dim=64, include_mhc=False, cache_dir=None)
    X, names = build_feature_matrix(toy_df.head(20), cfg)
    assert X.shape[0] == 20
    assert X.shape[1] == len(names)
    # Two fields (cdr3b, peptide) x 64 dims.
    assert X.shape[1] == 128


def test_cache_roundtrip(tmp_path):
    e = get_embedder(EmbeddingConfig(backend="fallback", fallback_dim=64))
    seqs = ["GILGFVFTL", "CASSF", "GILGFVFTL"]
    r1 = cached_embed(e, seqs, pooling="mean", cache_dir=tmp_path)
    r2 = cached_embed(e, seqs, pooling="mean", cache_dir=tmp_path)  # now from cache
    assert np.allclose(r1.vectors, r2.vectors)
    assert any(tmp_path.rglob("*.npy"))
