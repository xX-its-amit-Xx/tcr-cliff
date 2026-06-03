# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Source-agnostic loaders that normalise any dataset to the canonical schema."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from tcr_cliff._logging import get_logger
from tcr_cliff.config import Config, DataConfig
from tcr_cliff.data.schema import validate_pairs

_log = get_logger("data.loaders")

_TOY_PACKAGE = "tcr_cliff.data.toy"


def toy_path(name: str = "toy_pairs.csv") -> Path:
    """Return the filesystem path to a bundled toy resource."""
    return Path(str(resources.files(_TOY_PACKAGE).joinpath(name)))


def load_toy() -> pd.DataFrame:
    """Load the bundled synthetic toy pairs table (validated, normalised)."""
    df = pd.read_csv(toy_path("toy_pairs.csv"))
    return validate_pairs(df)


def load_toy_structure_summary() -> dict[str, dict]:
    """Load precomputed structure-confidence summaries for the toy set."""
    path = toy_path("structure/summary.json")
    return json.loads(Path(path).read_text())


def load_csv(path: str | Path, **read_kwargs) -> pd.DataFrame:
    """Load and validate a user CSV in the canonical schema."""
    df = pd.read_csv(path, **read_kwargs)
    return validate_pairs(df)


def load_pairs(cfg: Config | DataConfig) -> pd.DataFrame:
    """Dispatch to the right loader based on ``DataConfig.source``.

    For network sources (vdjdb/mcpas/iedb/nettcr) this delegates to
    :mod:`tcr_cliff.data.download`, which is imported lazily so the import graph
    stays light for offline use.
    """
    data_cfg = cfg.data if isinstance(cfg, Config) else cfg
    src = data_cfg.source
    if src == "toy":
        return load_toy()
    if src == "csv":
        if data_cfg.path is None:
            raise ValueError("DataConfig.source='csv' requires `path`")
        return load_csv(data_cfg.path)
    # Heavy/network sources are resolved lazily.
    from tcr_cliff.data import download

    loader = {
        "vdjdb": download.load_vdjdb,
        "mcpas": download.load_mcpas,
        "iedb": download.load_iedb,
        "nettcr": download.load_nettcr,
    }[src]
    return validate_pairs(loader(cache_dir=data_cfg.path))


def split_pairs(
    df: pd.DataFrame,
    cfg: DataConfig | None = None,
    *,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Split into train/val/test, honouring an existing split column or grouping key.

    Grouped splitting (default on ``peptide``) prevents the same peptide—and therefore
    a cliff pair's two halves—from straddling train and test, which would leak the
    very signal the benchmark measures.

    Returns
    -------
    dict
        ``{"train": ..., "val": ..., "test": ...}`` of DataFrames.
    """
    cfg = cfg or DataConfig()
    if cfg.split_column and cfg.split_column in df.columns:
        col = cfg.split_column
    elif "split" in df.columns and df["split"].astype(str).str.len().gt(0).any():
        col = "split"
    else:
        col = None

    if col is not None:
        norm = df[col].astype(str).str.lower().str.strip()
        return {
            "train": df.loc[norm.isin(("train", "0", ""))].reset_index(drop=True),
            "val": df.loc[norm.isin(("val", "valid", "validation", "1"))].reset_index(drop=True),
            "test": df.loc[norm.isin(("test", "2"))].reset_index(drop=True),
        }

    rng = np.random.default_rng(seed)
    if cfg.group_split_on != "none" and cfg.group_split_on in df.columns:
        groups = df[cfg.group_split_on].to_numpy()
        uniq = np.array(sorted(set(groups)))
        rng.shuffle(uniq)
        n = len(uniq)
        test_g = set(uniq[: int(cfg.test_size * n)])
        val_g = set(uniq[int(cfg.test_size * n) : int((cfg.test_size + cfg.val_size) * n)])
        which = np.where(
            np.isin(groups, list(test_g)),
            "test",
            np.where(np.isin(groups, list(val_g)), "val", "train"),
        )
    else:
        u = rng.random(len(df))
        which = np.where(
            u < cfg.test_size, "test", np.where(u < cfg.test_size + cfg.val_size, "val", "train")
        )
    out = {k: df.loc[which == k].reset_index(drop=True) for k in ("train", "val", "test")}
    _log.info(
        "split sizes: train=%d val=%d test=%d (grouped on %s)",
        len(out["train"]),
        len(out["val"]),
        len(out["test"]),
        cfg.group_split_on,
    )
    return out
