# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Matplotlib plotting helpers for interpretation outputs.

All plotting uses the non-interactive ``Agg`` backend and writes PNG files; no
figure is ever shown interactively (``plt.show`` is never called), so these
helpers are safe in headless/CI environments. Matplotlib is configured to Agg at
import time.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np

from tcr_cliff._logging import get_logger

_log = get_logger("interpret.plots")


def plot_residue_importance(
    attr: np.ndarray,
    sequence: str,
    path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Bar plot of per-residue importance, x-ticked by residue and position.

    Parameters
    ----------
    attr:
        Per-residue importance array of length ``len(sequence)``.
    sequence:
        The sequence the attribution maps onto (used for x tick labels).
    path:
        Output PNG path; parent directories are created if needed.
    title:
        Optional plot title.

    Returns
    -------
    pathlib.Path
        The path the figure was written to.

    Raises
    ------
    ValueError
        If ``len(attr) != len(sequence)``.
    """
    attr = np.asarray(attr, dtype=np.float64).reshape(-1)
    if attr.shape[0] != len(sequence):
        raise ValueError(f"attribution length {attr.shape[0]} != sequence length {len(sequence)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(4.0, 0.5 * len(sequence)), 3.0))
    positions = np.arange(len(sequence))
    colors = ["#c44e52" if v >= 0 else "#4c72b0" for v in attr]
    ax.bar(positions, attr, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{aa}\nP{i + 1}" for i, aa in enumerate(sequence)], fontsize=8)
    ax.set_ylabel("importance (score drop)")
    ax.set_title(title or "Per-residue importance")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    _log.info("wrote residue-importance plot to %s", path)
    return path


def plot_field_importance(
    field_scores: dict,
    path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Horizontal bar plot of per-field importance blocks.

    Parameters
    ----------
    field_scores:
        Mapping ``{field: importance}`` (e.g. the ``field_importance`` block of
        :func:`tcr_cliff.interpret.shap_baseline.explain_baseline`).
    path:
        Output PNG path; parent directories are created if needed.
    title:
        Optional plot title.

    Returns
    -------
    pathlib.Path
        The path the figure was written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    items = sorted(field_scores.items(), key=lambda kv: kv[1])
    fields = [k for k, _ in items]
    values = [float(v) for _, v in items]

    fig, ax = plt.subplots(figsize=(5.0, max(2.0, 0.6 * max(len(fields), 1))))
    ax.barh(range(len(fields)), values, color="#55a868")
    ax.set_yticks(range(len(fields)))
    ax.set_yticklabels(fields)
    ax.set_xlabel("aggregated importance")
    ax.set_title(title or "Per-field importance")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    _log.info("wrote field-importance plot to %s", path)
    return path


def plot_cliff_graph(
    g: object,
    path: str | Path,
    *,
    title: str | None = None,
    cliffs_only: bool = False,
    seed: int = 0,
) -> Path:
    """Draw a cliff graph, colouring cliff edges distinctly from smooth ones.

    Parameters
    ----------
    g:
        A :class:`networkx.Graph` produced by
        :func:`tcr_cliff.cliffs.build_cliff_graph`; edges may carry an
        ``is_cliff`` attribute.
    path:
        Output PNG path; parent directories are created if needed.
    title:
        Optional plot title.
    cliffs_only:
        If True, only cliff edges (``is_cliff``) are drawn.
    seed:
        Spring-layout random seed for reproducible node placement.

    Returns
    -------
    pathlib.Path
        The path the figure was written to.

    Raises
    ------
    ImportError
        If ``networkx`` is not installed.
    """
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - networkx is a spine dep
        raise ImportError(
            "plot_cliff_graph requires 'networkx'. Install it with " "`pip install networkx`."
        ) from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    edges = list(g.edges(data=True))
    if cliffs_only:
        edges = [(u, v, d) for u, v, d in edges if d.get("is_cliff")]
    cliff_edges = [(u, v) for u, v, d in edges if d.get("is_cliff")]
    smooth_edges = [(u, v) for u, v, d in edges if not d.get("is_cliff")]

    pos = nx.spring_layout(g, seed=seed)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=80, node_color="#dddddd")
    if smooth_edges:
        nx.draw_networkx_edges(
            g, pos, ax=ax, edgelist=smooth_edges, edge_color="#999999", width=1.0
        )
    if cliff_edges:
        nx.draw_networkx_edges(g, pos, ax=ax, edgelist=cliff_edges, edge_color="#c44e52", width=2.0)
    ax.set_title(title or "TCR-pMHC cliff graph")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    _log.info(
        "wrote cliff-graph plot to %s (%d cliff / %d smooth edges)",
        path,
        len(cliff_edges),
        len(smooth_edges),
    )
    return path
