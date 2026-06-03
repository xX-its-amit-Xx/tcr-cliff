# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Canonical schema for a TCR-pMHC binding table.

Every loader (toy, CSV, VDJdb, McPAS, IEDB, NetTCR) normalises to this column set
so that the rest of the pipeline is source-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

#: The 20 canonical amino acids.
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = frozenset(AA_ALPHABET)

#: Required columns in a normalised pairs table.
REQUIRED_COLUMNS: tuple[str, ...] = ("pair_id", "cdr3b", "peptide", "binder")

#: Optional but recognised columns.
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "mhc",
    "mhc_pseudo",
    "v_gene",
    "j_gene",
    "affinity",
    "source",
    "split",
)

ALL_COLUMNS: tuple[str, ...] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


@dataclass(frozen=True)
class SchemaError(Exception):
    """Raised when a pairs table does not satisfy the schema."""

    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


def clean_sequence(seq: str) -> str:
    """Upper-case and strip a protein sequence, returning ``''`` for missing values."""
    if seq is None or (isinstance(seq, float) and pd.isna(seq)):
        return ""
    return str(seq).strip().upper()


def is_valid_peptide(seq: str) -> bool:
    """True if every character is a canonical amino acid and the sequence is non-empty."""
    seq = clean_sequence(seq)
    return len(seq) > 0 and all(c in AA_SET for c in seq)


def validate_pairs(df: pd.DataFrame, *, require_affinity: bool = False) -> pd.DataFrame:
    """Validate and normalise a pairs DataFrame in place-ish (returns a copy).

    Ensures required columns exist, fills optional columns with sensible defaults,
    cleans sequences, and coerces the ``binder`` label to ``{0, 1}``.

    Raises
    ------
    SchemaError
        If a required column is missing or no valid rows remain.
    """
    df = df.copy()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"pairs table missing required columns: {missing}")

    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = _default_for(col)

    df["cdr3b"] = df["cdr3b"].map(clean_sequence)
    df["peptide"] = df["peptide"].map(clean_sequence)
    df["mhc_pseudo"] = df["mhc_pseudo"].map(clean_sequence)
    df["binder"] = df["binder"].astype("float").round().astype("int")
    if not df["binder"].isin((0, 1)).all():
        raise SchemaError("binder column must contain only 0/1 values")
    if require_affinity and df["affinity"].isna().all():
        raise SchemaError("affinity required but column is entirely missing")

    valid = df["cdr3b"].map(is_valid_peptide) & df["peptide"].map(is_valid_peptide)
    df = df.loc[valid].reset_index(drop=True)
    if df.empty:
        raise SchemaError("no valid rows after sequence validation")
    return df[list(ALL_COLUMNS)]


def _default_for(col: str):
    if col == "affinity":
        return float("nan")
    if col == "source":
        return "unknown"
    if col == "split":
        return ""
    return ""
