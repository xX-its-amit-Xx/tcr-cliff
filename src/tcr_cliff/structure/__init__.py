# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Structural-confidence ingestion for the fusion model.

This package turns AlphaFold/ESMFold confidence outputs into a fixed 12-dimensional
per-pair feature block. :func:`load_structure_features` is the single entry point the
model code calls; it either reads the bundled toy summary (``use_stub``) or parses
per-pair JSON confidence files from ``source_dir``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tcr_cliff._logging import get_logger
from tcr_cliff.config import StructureConfig
from tcr_cliff.structure.features import (
    STRUCTURE_FEATURE_NAMES,
    features_from_record,
    features_from_summary,
    missing_vector,
)
from tcr_cliff.structure.parse import (
    ConfidenceRecord,
    parse_af_json,
    parse_pdb_plddt,
)

_log = get_logger("structure")

__all__ = [
    "STRUCTURE_FEATURE_NAMES",
    "ConfidenceRecord",
    "features_from_record",
    "features_from_summary",
    "load_structure_features",
    "missing_vector",
    "parse_af_json",
    "parse_pdb_plddt",
]


def load_structure_features(
    df: pd.DataFrame, structure_cfg: StructureConfig
) -> tuple[np.ndarray, list[str]]:
    """Assemble the per-pair structural-confidence feature matrix.

    Two modes:

    * ``structure_cfg.use_stub`` (default): look each ``pair_id`` up in the bundled
      toy structure summary (:func:`tcr_cliff.data.load_toy_structure_summary`).
    * ``structure_cfg.source_dir`` set: read ``<source_dir>/<pair_id>.json`` for each
      row via :func:`parse_af_json` and reduce it with :func:`features_from_record`,
      using ``structure_cfg.pae_interface_cutoff`` as the interface PAE threshold.

    Rows with no matching entry (or an unreadable file) receive the missing vector
    (all zeros, ``available_flag=0``).

    Parameters
    ----------
    df:
        Normalised pairs frame; must contain a ``pair_id`` column.
    structure_cfg:
        Structure configuration controlling the source.

    Returns
    -------
    tuple of (np.ndarray, list of str)
        ``X`` of shape ``[N, 12]`` (float32) and the 12 feature names.
    """
    n = len(df)
    names = list(STRUCTURE_FEATURE_NAMES)
    pair_ids = df["pair_id"].astype(str).tolist() if "pair_id" in df.columns else [""] * n

    source_dir = structure_cfg.source_dir
    use_files = source_dir is not None and not structure_cfg.use_stub

    rows: list[np.ndarray] = []
    n_present = 0

    if use_files:
        from pathlib import Path

        base = Path(source_dir)
        cutoff = float(structure_cfg.pae_interface_cutoff)
        for pid in pair_ids:
            json_path = base / f"{pid}.json"
            if json_path.exists():
                try:
                    rec = parse_af_json(json_path)
                    rows.append(features_from_record(rec, cutoff=cutoff))
                    n_present += 1
                    continue
                except (OSError, ValueError, KeyError) as exc:
                    _log.warning("failed to parse %s: %s", json_path, exc)
            rows.append(missing_vector())
    else:
        # Stub / precomputed summary path.
        from tcr_cliff.data import load_toy_structure_summary

        summary = load_toy_structure_summary()
        for pid in pair_ids:
            entry = summary.get(pid)
            if entry is not None:
                rows.append(features_from_summary(entry))
                n_present += 1
            else:
                rows.append(missing_vector())

    X = np.vstack(rows).astype(np.float32) if rows else np.zeros((0, len(names)), dtype=np.float32)

    _log.info(
        "structure features: %d/%d pairs with confidence (%s)",
        n_present,
        n,
        "files" if use_files else "stub summary",
    )
    return X, names
