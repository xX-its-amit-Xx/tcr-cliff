# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Map per-residue attribution onto canonical peptide anchor positions.

For MHC class I, the dominant anchor residues are typically **P2** (the second
residue) and **PΩ** (the C-terminal residue). These engage MHC binding pockets,
while the central "bulge" residues are the primary TCR-contact residues. This
module translates a residue-attribution vector into anchor-aware summaries so an
analyst can quickly see whether a prediction is driven by MHC-anchoring chemistry
or by TCR-facing recognition residues.
"""

from __future__ import annotations

import numpy as np

from tcr_cliff._logging import get_logger

_log = get_logger("interpret.anchors")


def map_to_positions(attr: np.ndarray, sequence: str) -> list[tuple[int, str, float]]:
    """Zip an attribution vector with 1-based positions and residues.

    Parameters
    ----------
    attr:
        Per-residue importance array of length ``len(sequence)``.
    sequence:
        The sequence the attribution was computed over.

    Returns
    -------
    list of (int, str, float)
        ``(position_1based, residue, importance)`` tuples in sequence order.

    Raises
    ------
    ValueError
        If ``len(attr) != len(sequence)``.
    """
    attr = np.asarray(attr, dtype=np.float64).reshape(-1)
    if attr.shape[0] != len(sequence):
        raise ValueError(f"attribution length {attr.shape[0]} != sequence length {len(sequence)}")
    return [(i + 1, sequence[i], float(attr[i])) for i in range(len(sequence))]


def anchor_importance(
    attr: np.ndarray, peptide: str, *, anchor_positions: tuple[int, ...] = (2,)
) -> dict:
    """Summarise attribution at canonical MHC-I anchor positions (P2, PΩ).

    Anchors are defined as the residues at ``anchor_positions`` (1-based; P2 by
    default) plus the C-terminal residue PΩ. The function reports each anchor's
    importance and whether anchors dominate the attribution mass relative to the
    non-anchor (TCR-contact) residues.

    Parameters
    ----------
    attr:
        Per-residue importance array, length ``len(peptide)``.
    peptide:
        The peptide sequence the attribution was computed over.
    anchor_positions:
        Additional fixed 1-based anchor positions besides PΩ. Defaults to
        ``(2,)`` so the canonical anchor set is ``{P2, PΩ}``.

    Returns
    -------
    dict
        Keys:

        ``anchors``
            ``{label: {"position": int, "residue": str, "importance": float}}``
            for each resolved anchor (e.g. ``"P2"``, ``"POmega"``).
        ``anchor_importance_sum`` / ``non_anchor_importance_sum``
            Summed absolute importance over anchor vs non-anchor residues.
        ``anchor_fraction``
            Fraction of total absolute importance located on anchors (0..1).
        ``anchors_dominate``
            ``True`` when anchors carry > 50% of total absolute importance.
        ``length``
            Peptide length.

    Raises
    ------
    ValueError
        If ``len(attr) != len(peptide)``.
    """
    attr = np.asarray(attr, dtype=np.float64).reshape(-1)
    length = len(peptide)
    if attr.shape[0] != length:
        raise ValueError(f"attribution length {attr.shape[0]} != peptide length {length}")
    if length == 0:
        return {
            "anchors": {},
            "anchor_importance_sum": 0.0,
            "non_anchor_importance_sum": 0.0,
            "anchor_fraction": 0.0,
            "anchors_dominate": False,
            "length": 0,
        }

    abs_attr = np.abs(attr)
    anchor_idx: dict[str, int] = {}
    for pos in anchor_positions:
        if 1 <= pos <= length:
            anchor_idx[f"P{pos}"] = pos - 1
    anchor_idx["POmega"] = length - 1  # C-terminal anchor (PΩ)

    anchors: dict[str, dict] = {}
    for label, idx in anchor_idx.items():
        anchors[label] = {
            "position": idx + 1,
            "residue": peptide[idx],
            "importance": float(attr[idx]),
        }

    unique_idx = set(anchor_idx.values())
    anchor_sum = float(abs_attr[list(unique_idx)].sum()) if unique_idx else 0.0
    total = float(abs_attr.sum())
    non_anchor_sum = total - anchor_sum
    fraction = anchor_sum / total if total > 0 else 0.0

    result = {
        "anchors": anchors,
        "anchor_importance_sum": anchor_sum,
        "non_anchor_importance_sum": non_anchor_sum,
        "anchor_fraction": fraction,
        "anchors_dominate": fraction > 0.5,
        "length": length,
    }
    _log.debug(
        "anchor_importance: L=%d anchor_fraction=%.3f dominate=%s",
        length,
        fraction,
        result["anchors_dominate"],
    )
    return result
