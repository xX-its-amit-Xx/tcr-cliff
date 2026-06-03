# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Build and analyse the cliff-pair graph with networkx.

Nodes are TCR-pMHC records; edges connect within-``k`` neighbours and are tagged
``cliff`` or ``smooth``. The graph view exposes structure that flat pair lists hide:
cliff hubs (peptides at the centre of many opposing outcomes), connected components
of mutually-reachable variants, and the local cliff density around any record.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd

from tcr_cliff._logging import get_logger
from tcr_cliff.cliffs.detect import NeighborPair

_log = get_logger("cliffs.graph")


def build_cliff_graph(
    pairs: list[NeighborPair],
    df: pd.DataFrame | None = None,
    *,
    cliffs_only: bool = False,
) -> nx.Graph:
    """Construct a networkx graph from neighbour pairs.

    Parameters
    ----------
    pairs:
        Neighbour pairs from :func:`tcr_cliff.cliffs.detect.find_neighbor_pairs`.
    df:
        Optional source table; if given, node attributes (peptide, cdr3b, mhc,
        binder) are attached for downstream plotting/inspection.
    cliffs_only:
        If True, omit smooth-neighbour edges.

    Returns
    -------
    networkx.Graph
        Undirected graph; edges carry ``vary``, ``distance``, ``is_cliff``,
        ``reason`` attributes.
    """
    g = nx.Graph()
    if df is not None:
        id_col = "pair_id" if "pair_id" in df.columns else None
        for idx, row in df.iterrows():
            node = row["pair_id"] if id_col else idx
            g.add_node(
                node,
                peptide=row.get("peptide", ""),
                cdr3b=row.get("cdr3b", ""),
                mhc=row.get("mhc", ""),
                binder=int(row.get("binder", 0)),
            )
    for p in pairs:
        if cliffs_only and not p.is_cliff:
            continue
        g.add_edge(
            p.id_i,
            p.id_j,
            vary=p.vary,
            distance=p.distance,
            is_cliff=p.is_cliff,
            reason=p.reason,
        )
    _log.info(
        "cliff graph: %d nodes, %d edges (cliffs_only=%s)",
        g.number_of_nodes(),
        g.number_of_edges(),
        cliffs_only,
    )
    return g


def cliff_components(g: nx.Graph, *, cliffs_only: bool = True) -> list[set]:
    """Return connected components over (optionally cliff-only) edges, largest first."""
    if cliffs_only:
        sub = g.edge_subgraph(
            [(u, v) for u, v, d in g.edges(data=True) if d.get("is_cliff")]
        ).copy()
    else:
        sub = g
    return sorted(nx.connected_components(sub), key=len, reverse=True)


def cliff_hubs(g: nx.Graph, top_n: int = 10) -> list[tuple]:
    """Rank nodes by cliff degree (number of incident cliff edges)."""
    deg: dict = {}
    for u, v, d in g.edges(data=True):
        if d.get("is_cliff"):
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
    return sorted(deg.items(), key=lambda kv: kv[1], reverse=True)[:top_n]


def export_graph(g: nx.Graph, path: str) -> str:
    """Write the graph to GraphML (interoperable with Gephi/Cytoscape)."""
    nx.write_graphml(g, path)
    _log.info("wrote cliff graph to %s", path)
    return path
