# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Typer command-line interface for the tcr-cliff pipeline.

Five subcommands wrap the library so an end-to-end run is reproducible from one
config file:

* ``embed``        - build the embedding feature matrix and dump it to disk.
* ``find-cliffs``  - mine within-``k`` neighbour pairs, write the cliff table,
  statistics, and the cliff graph.
* ``train``        - load + split data, train the configured model, save it, and
  emit test-set predictions with cliff annotations.
* ``eval``         - score a saved model on held-out data and write the headline
  cliff-aware report.
* ``explain``      - SHAP + per-residue attribution for a single record, with plots.

Every command accepts an optional ``--config`` YAML plus the common ``--data``,
``--output``, and ``--seed`` overrides. User-facing progress is emitted with
:func:`typer.echo`; everything else logs through :mod:`tcr_cliff._logging`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from tcr_cliff._logging import get_logger, setup_logging
from tcr_cliff.config import Config, load_config
from tcr_cliff.utils.seed import seed_everything

_log = get_logger("cli")

app = typer.Typer(
    name="tcr-cliff",
    add_completion=False,
    no_args_is_help=True,
    help="Activity-cliff-aware TCR-pMHC binding prediction.",
)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _prepare(
    config: Path | None,
    data: str | None,
    output: Path | None,
    seed: int | None,
) -> Config:
    """Load a :class:`Config`, apply CLI overrides, and seed + configure logging.

    Parameters
    ----------
    config:
        Path to a YAML config file, or ``None`` to use the package defaults.
    data:
        Optional override for ``cfg.data.source``. A value containing a path
        separator or ending in ``.csv`` is treated as a CSV file and sets
        ``cfg.data.source='csv'`` with ``cfg.data.path``.
    output:
        Optional override for ``cfg.output_dir``.
    seed:
        Optional override for ``cfg.seed`` (also propagated to ``cfg.train.seed``).

    Returns
    -------
    Config
        The fully resolved configuration with logging and RNGs initialised.
    """
    cfg = load_config(config) if config is not None else Config()

    if data is not None:
        _apply_data_override(cfg, data)
    if output is not None:
        cfg.output_dir = Path(output)
    if seed is not None:
        cfg.seed = int(seed)
        cfg.train.seed = int(seed)

    setup_logging()
    seed_everything(cfg.seed)
    cfg.output_dir = Path(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _log.info(
        "resolved config: source=%s output_dir=%s seed=%d",
        cfg.data.source,
        cfg.output_dir,
        cfg.seed,
    )
    return cfg


def _apply_data_override(cfg: Config, data: str) -> None:
    """Interpret the ``--data`` value as a known source name or a CSV path."""
    known = {"toy", "csv", "vdjdb", "mcpas", "iedb", "nettcr"}
    looks_like_path = data.endswith(".csv") or ("/" in data) or ("\\" in data)
    if data in known and not looks_like_path:
        cfg.data.source = data  # type: ignore[assignment]
    else:
        cfg.data.source = "csv"
        cfg.data.path = Path(data)


def _load_data(cfg: Config):
    """Load the pairs table selected by ``cfg.data`` (lazy import of loaders)."""
    from tcr_cliff.data import load_pairs

    df = load_pairs(cfg)
    _log.info("loaded %d pairs from source=%s", len(df), cfg.data.source)
    return df


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
@app.command()
def embed(
    config: Path | None = typer.Option(None, "--config", "-c", help="YAML config path."),
    data: str | None = typer.Option(None, "--data", help="Data source name or CSV path."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output directory."),
    seed: int | None = typer.Option(None, "--seed", help="Random seed override."),
) -> None:
    """Build the embedding feature matrix and save it as ``.npy`` + column names."""
    cfg = _prepare(config, data, output, seed)
    from tcr_cliff.embeddings import build_feature_matrix

    df = _load_data(cfg)
    X, names = build_feature_matrix(df, cfg.embedding)
    out = cfg.output_dir
    np.save(out / "features.npy", X)
    (out / "feature_names.json").write_text(json.dumps(names, indent=2))
    cfg.to_yaml(out / "config.yaml")
    typer.echo(f"embedded {X.shape[0]} rows -> {X.shape[1]} features at {out / 'features.npy'}")


@app.command("find-cliffs")
def find_cliffs_cmd(
    config: Path | None = typer.Option(None, "--config", "-c", help="YAML config path."),
    data: str | None = typer.Option(None, "--data", help="Data source name or CSV path."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output directory."),
    seed: int | None = typer.Option(None, "--seed", help="Random seed override."),
) -> None:
    """Mine neighbour/cliff pairs and write the table, statistics, and graph."""
    cfg = _prepare(config, data, output, seed)
    from tcr_cliff.cliffs import (
        build_cliff_graph,
        cliff_pairs_to_frame,
        cliff_statistics,
        export_graph,
        find_neighbor_pairs,
    )

    df = _load_data(cfg)
    pairs = find_neighbor_pairs(df, cfg.cliff)
    out = cfg.output_dir

    frame = cliff_pairs_to_frame(pairs)
    csv_path = out / "cliffs.csv"
    frame.to_csv(csv_path, index=False)

    stats = cliff_statistics(pairs, len(df))
    (out / "cliff_stats.json").write_text(json.dumps(stats, indent=2, default=float))

    graph = build_cliff_graph(pairs, df)
    export_graph(graph, str(out / "cliff_graph.graphml"))

    n_cliff = int(sum(p.is_cliff for p in pairs))
    typer.echo(f"found {len(pairs)} neighbour pairs ({n_cliff} cliffs) -> {csv_path}")


@app.command()
def train(
    config: Path | None = typer.Option(None, "--config", "-c", help="YAML config path."),
    data: str | None = typer.Option(None, "--data", help="Data source name or CSV path."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output directory."),
    seed: int | None = typer.Option(None, "--seed", help="Random seed override."),
) -> None:
    """Train the configured model, save it, and write test-set predictions."""
    cfg = _prepare(config, data, output, seed)
    from tcr_cliff.cliffs import find_neighbor_pairs, mark_cliff_membership
    from tcr_cliff.data import split_pairs
    from tcr_cliff.models import predict_scores, train_model

    df = _load_data(cfg)
    splits = split_pairs(df, cfg.data, seed=cfg.seed)
    df_train, df_val, df_test = splits["train"], splits["val"], splits["test"]
    if df_train.empty:
        raise typer.BadParameter("training split is empty; check data/split settings")

    model = train_model(cfg, df_train, df_val if len(df_val) else None)
    out = cfg.output_dir
    model_dir = model.save(out / "model")
    cfg.to_yaml(out / "config.yaml")

    if len(df_test):
        df_test = df_test.reset_index(drop=True)
        scores = predict_scores(model, df_test)
        pairs = find_neighbor_pairs(df_test, cfg.cliff)
        marked = mark_cliff_membership(df_test, pairs)
        preds = marked[["pair_id", "binder", "in_cliff", "in_neighbor"]].copy()
        preds["score"] = np.asarray(scores, dtype=float)
        preds.to_csv(out / "predictions.csv", index=False)
        typer.echo(f"trained {model.kind} -> {model_dir}; wrote {len(preds)} test predictions")
    else:
        typer.echo(f"trained {model.kind} -> {model_dir}; no test split to predict")


@app.command()
def eval(
    config: Path | None = typer.Option(None, "--config", "-c", help="YAML config path."),
    data: str | None = typer.Option(None, "--data", help="Data source name or CSV path."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output directory."),
    seed: int | None = typer.Option(None, "--seed", help="Random seed override."),
    model_path: Path | None = typer.Option(
        None, "--model", "-m", help="Saved model directory (default: <output>/model)."
    ),
) -> None:
    """Score a saved model on held-out data and write the cliff-aware report."""
    cfg = _prepare(config, data, output, seed)
    from tcr_cliff.data import split_pairs
    from tcr_cliff.eval import (
        cliff_aware_report,
        cliff_report_to_frame,
        save_report,
    )
    from tcr_cliff.models import load_model, predict_scores

    mpath = Path(model_path) if model_path is not None else cfg.output_dir / "model"
    if not mpath.exists():
        raise typer.BadParameter(f"model directory not found: {mpath}")
    model = load_model(mpath)

    df = _load_data(cfg)
    df_test = split_pairs(df, cfg.data, seed=cfg.seed)["test"].reset_index(drop=True)
    if df_test.empty:
        df_test = df.reset_index(drop=True)
        _log.warning("empty test split; evaluating on the full table")

    scores = predict_scores(model, df_test)
    report = cliff_aware_report(df_test, scores, cliff_cfg=cfg.cliff, threshold=cfg.eval.threshold)
    out = cfg.output_dir
    save_report(report, out / "eval_report.json")

    frame = cliff_report_to_frame(report)
    typer.echo(frame.to_string(index=False))
    overall = report["overall"].get("auroc", float("nan"))
    cliff_auroc = report["cliff_records"].get("auroc", float("nan"))
    typer.echo(f"overall AUROC={overall:.3f} | cliff-record AUROC={cliff_auroc:.3f}")


@app.command()
def explain(
    config: Path | None = typer.Option(None, "--config", "-c", help="YAML config path."),
    data: str | None = typer.Option(None, "--data", help="Data source name or CSV path."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output directory."),
    seed: int | None = typer.Option(None, "--seed", help="Random seed override."),
    model_path: Path | None = typer.Option(
        None, "--model", "-m", help="Saved baseline model directory (default: <output>/model)."
    ),
    row: int = typer.Option(0, "--row", help="Positional row index to attribute."),
    field: str = typer.Option("peptide", "--field", help="Sequence field to attribute over."),
) -> None:
    """Explain a baseline model: SHAP blocks + per-residue attribution with plots."""
    cfg = _prepare(config, data, output, seed)
    # interpret is a leaf module; import lazily so the CLI imports without it.
    from tcr_cliff.interpret import (
        explain_baseline,
        plot_field_importance,
        plot_residue_importance,
        residue_attribution,
    )
    from tcr_cliff.models import load_model, predict_scores

    mpath = Path(model_path) if model_path is not None else cfg.output_dir / "model"
    if not mpath.exists():
        raise typer.BadParameter(f"model directory not found: {mpath}")
    model = load_model(mpath)

    df = _load_data(cfg).reset_index(drop=True)
    out = cfg.output_dir

    field_scores = explain_baseline(model, df)
    (out / "shap_fields.json").write_text(json.dumps(field_scores, indent=2, default=float))
    plot_field_importance(field_scores, str(out / "field_importance.png"))

    record = df.iloc[int(row)]
    attr = residue_attribution(lambda d: predict_scores(model, d), record, field=field, cfg=cfg)
    seq = str(record[field])
    plot_residue_importance(attr, seq, str(out / "residue_importance.png"))
    np.save(out / "residue_attribution.npy", np.asarray(attr, dtype=float))
    typer.echo(f"explained row {row} field={field} ({seq}); plots + SHAP blocks in {out}")


def main() -> None:
    """Console-script entry point (``tcr-cliff = tcr_cliff.cli.main:main``)."""
    app()


if __name__ == "__main__":  # pragma: no cover - module executed as a script
    main()
