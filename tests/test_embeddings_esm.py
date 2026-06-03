# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Tests for the ESM-2 embedders.

The lightweight tests here exercise pure-Python behaviour (HF-id mapping, cache
keys, constructor validation, lazy loading) and never import torch / fair-esm /
transformers. The real-model tests are gated behind ``pytest.importorskip`` and
``@pytest.mark.esm`` so they skip in the offline suite (fair-esm is absent here,
and we deliberately avoid downloading HuggingFace weights).
"""

from __future__ import annotations

import numpy as np
import pytest

from tcr_cliff.embeddings.esm import ESMEmbedder, _to_hf_id


# --------------------------------------------------------------------------- #
# Lightweight tests: no heavy imports, safe to run anywhere.
# --------------------------------------------------------------------------- #
def test_to_hf_id_maps_fair_names():
    assert _to_hf_id("esm2_t12_35M_UR50D") == "facebook/esm2_t12_35M_UR50D"


def test_to_hf_id_passes_through_qualified_ids():
    assert _to_hf_id("facebook/esm2_t6_8M_UR50D") == "facebook/esm2_t6_8M_UR50D"
    assert _to_hf_id("some-org/custom-esm") == "some-org/custom-esm"


def test_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown ESM backend"):
        ESMEmbedder(backend="not-a-backend")


def test_name_is_filesystem_safe_cache_key():
    emb = ESMEmbedder(model_name="esm2_t12_35M_UR50D", backend="esm-hf", layers=(-1,))
    # The cache layer uses ``name`` as a directory, so it must avoid brackets/commas.
    assert emb.name == "esm-esm-hf-esm2_t12_35M_UR50D-L-1"
    for bad in "[](), ":
        assert bad not in emb.name


def test_name_encodes_multiple_layers():
    emb = ESMEmbedder(backend="esm-fair", layers=(-1, -2, 6))
    assert emb.name == "esm-esm-fair-esm2_t12_35M_UR50D-L-1_-2_6"
    assert emb.layers == (-1, -2, 6)


def test_constructor_stores_params_without_loading_model():
    emb = ESMEmbedder(
        model_name="esm2_t6_8M_UR50D",
        backend="esm-hf",
        device="cpu",
        layers=(-1,),
        batch_size=4,
        max_len=32,
    )
    assert emb.model_name == "esm2_t6_8M_UR50D"
    assert emb.backend == "esm-hf"
    assert emb.device == "cpu"
    assert emb.batch_size == 4
    assert emb.max_len == 32
    # Nothing should be loaded merely by constructing the embedder.
    assert emb._model is None
    assert emb._dim is None


def test_prep_uppercases_and_truncates():
    emb = ESMEmbedder(max_len=4)
    assert emb._prep("gilg") == "GILG"
    assert emb._prep("GILGFVFTL") == "GILG"  # truncated to max_len
    assert emb._prep("") == ""
    assert emb._prep(None) == ""  # type: ignore[arg-type]


def test_batched_chunks_items():
    emb = ESMEmbedder(batch_size=2)
    chunks = list(emb._batched(["a", "b", "c", "d", "e"], 2))
    assert [start for start, _ in chunks] == [0, 2, 4]
    assert [c for _, c in chunks] == [["a", "b"], ["c", "d"], ["e"]]


def test_abs_layers_translates_negative_indices():
    emb = ESMEmbedder(layers=(-1, -2, 0))
    emb._n_layers = 12  # pretend a 12-layer model is loaded
    # hidden_states has length n_layers + 1 (index 0 = embeddings, 12 = final).
    assert emb._abs_layers() == [12, 11, 0]


# --------------------------------------------------------------------------- #
# Real-model tests: skipped offline (no fair-esm; no HF weight downloads).
# --------------------------------------------------------------------------- #
@pytest.mark.esm
@pytest.mark.slow
def test_fair_backend_mean_pool_shape():
    pytest.importorskip("esm")
    pytest.importorskip("torch")
    emb = ESMEmbedder(model_name="esm2_t6_8M_UR50D", backend="esm-fair", batch_size=2)
    res = emb.embed(["GILGFVFTL", "CASSF"], pooling="mean")
    assert res.vectors.shape == (2, emb.dim)
    assert np.isfinite(res.vectors).all()


@pytest.mark.esm
@pytest.mark.slow
def test_fair_backend_per_residue_shape():
    pytest.importorskip("esm")
    pytest.importorskip("torch")
    emb = ESMEmbedder(model_name="esm2_t6_8M_UR50D", backend="esm-fair", batch_size=2)
    res = emb.embed(["GILGFVFTL", "CASSF"], pooling="per_residue")
    assert len(res.per_residue) == 2
    # BOS/EOS excluded => one row per residue.
    assert res.per_residue[0].shape[0] == 9
    assert res.per_residue[1].shape[0] == 5
    assert res.per_residue[0].shape[1] == emb.dim


@pytest.mark.esm
@pytest.mark.slow
def test_hf_backend_mean_pool_shape():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    # NOTE: this downloads weights on first run; kept behind 'slow'+'esm' so the
    # offline suite never triggers a network fetch.
    emb = ESMEmbedder(model_name="esm2_t6_8M_UR50D", backend="esm-hf", batch_size=2)
    res = emb.embed(["GILGFVFTL", "CASSF"], pooling="mean")
    assert res.vectors.shape == (2, emb.dim)
    assert np.isfinite(res.vectors).all()
