# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""End-to-end CLI tests using Typer's :class:`CliRunner` on the toy data.

These stay fast and torch-free: the baseline path uses the fallback embedder with a
small ``fallback_dim`` and a small number of boosting estimators. They exercise the
three core commands that compose a full run — ``find-cliffs``, ``train``, ``eval`` —
and assert that each writes its expected artifacts and exits cleanly.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from tcr_cliff.cli.main import app
from tcr_cliff.config import Config

runner = CliRunner()


def _baseline_config(tmp_path: Path) -> Path:
    """Write a tiny, offline-friendly baseline config and return its path."""
    cfg = Config()
    cfg.data.source = "toy"
    cfg.embedding.backend = "fallback"
    cfg.embedding.fallback_dim = 48
    cfg.embedding.include_mhc = False
    cfg.embedding.cache_dir = None
    cfg.model.kind = "baseline_lgbm"
    cfg.model.baseline.n_estimators = 40
    cfg.model.baseline.num_leaves = 7
    cfg.output_dir = tmp_path / "run"
    cfg.seed = 0
    path = tmp_path / "baseline.yaml"
    cfg.to_yaml(path)
    return path


def test_find_cliffs_writes_artifacts(tmp_path):
    cfg_path = _baseline_config(tmp_path)
    out = tmp_path / "cliffs_out"
    result = runner.invoke(
        app,
        ["find-cliffs", "--config", str(cfg_path), "--data", "toy", "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert (out / "cliffs.csv").exists()
    assert (out / "cliff_stats.json").exists()
    assert (out / "cliff_graph.graphml").exists()


def test_train_then_eval(tmp_path):
    cfg_path = _baseline_config(tmp_path)
    out = tmp_path / "model_out"

    train_res = runner.invoke(
        app,
        ["train", "--config", str(cfg_path), "--data", "toy", "--output", str(out), "--seed", "0"],
    )
    assert train_res.exit_code == 0, train_res.output
    assert (out / "model" / "meta.json").exists()
    assert (out / "predictions.csv").exists()

    eval_res = runner.invoke(
        app,
        ["eval", "--config", str(cfg_path), "--data", "toy", "--output", str(out)],
    )
    assert eval_res.exit_code == 0, eval_res.output
    assert (out / "eval_report.json").exists()


def test_embed_writes_feature_matrix(tmp_path):
    cfg_path = _baseline_config(tmp_path)
    out = tmp_path / "embed_out"
    result = runner.invoke(
        app,
        ["embed", "--config", str(cfg_path), "--data", "toy", "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert (out / "features.npy").exists()
    assert (out / "feature_names.json").exists()


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    # no_args_is_help exits with code 0 and prints usage including the command names.
    assert "find-cliffs" in result.output
    assert "train" in result.output
    assert "eval" in result.output
