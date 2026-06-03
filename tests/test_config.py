# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Tests for configuration loading and round-tripping."""

from __future__ import annotations

from tcr_cliff.config import Config, EmbeddingConfig, load_config


def test_defaults():
    cfg = Config()
    assert cfg.embedding.backend == "fallback"
    assert cfg.cliff.max_edits == 1
    assert cfg.model.kind == "baseline_lgbm"


def test_embedding_fields_toggle_mhc():
    assert EmbeddingConfig(include_mhc=True).fields == ("cdr3b", "peptide", "mhc_pseudo")
    assert EmbeddingConfig(include_mhc=False).fields == ("cdr3b", "peptide")


def test_seed_propagates():
    cfg = Config(seed=42)
    assert cfg.train.seed == 42


def test_yaml_roundtrip(tmp_path):
    cfg = Config(seed=7)
    cfg.cliff.max_edits = 2
    path = cfg.to_yaml(tmp_path / "cfg.yaml")
    loaded = load_config(path)
    assert loaded.seed == 7
    assert loaded.cliff.max_edits == 2


def test_load_config_overrides():
    cfg = load_config(None, seed=11)
    assert cfg.seed == 11
