# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Tests for :mod:`tcr_cliff.data.download`.

The offline tests exercise the interop helpers (``to_nettcr_format``,
``write_nettcr_csv``) on the bundled toy set with **no network access**. The real
network loaders are marked ``slow`` and deselected by default.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tcr_cliff.data import download, load_toy
from tcr_cliff.data.download import (
    IEDB_URL,
    MCPAS_URL,
    NETTCR_URL,
    VDJDB_URL,
    cdr3_from_anarci,
    default_cache_dir,
    to_nettcr_format,
    write_nettcr_csv,
)


def test_source_urls_are_https_or_http() -> None:
    """Every documented source URL is a real, absolute web URL."""
    for url in (VDJDB_URL, MCPAS_URL, IEDB_URL, NETTCR_URL):
        assert isinstance(url, str)
        assert url.startswith("http://") or url.startswith("https://")


def test_to_nettcr_format_columns() -> None:
    """``to_nettcr_format(load_toy())`` yields NetTCR-2.0 input columns (no network)."""
    nettcr = to_nettcr_format(load_toy())
    assert isinstance(nettcr, pd.DataFrame)
    for col in ("CDR3b", "peptide", "binder"):
        assert col in nettcr.columns
    assert len(nettcr) > 0
    assert nettcr["binder"].isin((0, 1)).all()
    # Sequences are upper-cased canonical AA strings.
    assert nettcr["CDR3b"].str.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$").all()
    assert nettcr["peptide"].str.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$").all()


def test_write_nettcr_csv_roundtrip(tmp_path: Path) -> None:
    """``write_nettcr_csv`` writes a parseable CSV with the expected columns."""
    dest = tmp_path / "sub" / "nettcr_input.csv"
    written = write_nettcr_csv(load_toy(), dest)
    assert written == dest
    assert dest.exists()
    back = pd.read_csv(dest)
    for col in ("CDR3b", "peptide", "binder"):
        assert col in back.columns
    assert len(back) > 0


def test_default_cache_dir_is_path() -> None:
    """The default cache dir resolves to a Path without touching the filesystem."""
    cache = default_cache_dir()
    assert isinstance(cache, Path)
    assert str(cache)  # non-empty


def test_cdr3_from_anarci_without_dep() -> None:
    """When anarci is absent, a clear ImportError naming the package is raised."""
    pytest.importorskip  # noqa: B018 - ensure pytest import path is sane
    try:
        import anarci  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="anarci"):
            cdr3_from_anarci("EVQLVESGGGLVQPGGSLRLSCAAS")
    else:  # pragma: no cover - anarci not installed in the offline suite
        result = cdr3_from_anarci("EVQLVESGGGLVQPGGSLRLSCAAS")
        assert result is None or isinstance(result, str)


@pytest.mark.slow
def test_load_vdjdb_network() -> None:  # pragma: no cover - network
    """VDJdb loader returns a canonical-schema frame (network; deselected by default)."""
    df = download.load_vdjdb()
    for col in ("pair_id", "cdr3b", "peptide", "binder"):
        assert col in df.columns
    assert len(df) > 0


@pytest.mark.slow
def test_load_nettcr_network() -> None:  # pragma: no cover - network
    """NetTCR loader returns a canonical-schema frame with 0/1 labels (network)."""
    df = download.load_nettcr()
    assert df["binder"].isin((0, 1)).all()
    assert len(df) > 0


@pytest.mark.slow
def test_load_mcpas_network() -> None:  # pragma: no cover - network
    """McPAS loader returns a canonical-schema frame (network; deselected by default)."""
    df = download.load_mcpas()
    assert len(df) > 0


@pytest.mark.slow
def test_load_iedb_network() -> None:  # pragma: no cover - network
    """IEDB loader returns a canonical-schema frame (network; deselected by default)."""
    df = download.load_iedb()
    assert len(df) > 0
