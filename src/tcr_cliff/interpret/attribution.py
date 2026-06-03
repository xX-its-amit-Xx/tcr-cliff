# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Per-residue attribution for TCR-pMHC binding predictions.

The headline method, :func:`residue_attribution`, is **model-agnostic occlusion**:
it substitutes a neutral residue (alanine) at each position of a chosen sequence
field, re-scores the single record, and reports the drop in predicted score. It
only needs a ``predict_fn(df) -> scores`` callable, so it works with the
LightGBM/sklearn baseline (no torch required) and with any torch model alike.

For torch models that expose attention, :func:`attention_rollout` provides an
optional, lazily-imported alternative.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from tcr_cliff._logging import get_logger

_log = get_logger("interpret.attribution")

#: Default neutral substitution residue (small, non-polar, side-chain "blank").
NEUTRAL_RESIDUE = "A"


def residue_attribution(
    predict_fn: Callable[[pd.DataFrame], np.ndarray | Sequence[float]],
    row: pd.Series,
    *,
    field: str = "peptide",
    cfg: object | None = None,
    neutral: str = NEUTRAL_RESIDUE,
    relative: bool = False,
) -> np.ndarray:
    """Occlusion-based per-residue importance for one record.

    For each position ``i`` of ``row[field]``, the residue is replaced by
    ``neutral`` (alanine by default; a different residue is auto-selected when the
    native residue is already alanine, so importance is never trivially zero), the
    mutated single-row frame is scored with ``predict_fn``, and the **drop**
    ``base_score - mutated_score`` is recorded. Positive values mean the original
    residue *supported* the prediction.

    This is fully model-agnostic: ``predict_fn`` may wrap the baseline, a torch
    model, or any callable mapping a pairs ``DataFrame`` to per-row scores.

    Parameters
    ----------
    predict_fn:
        Callable mapping a pairs ``DataFrame`` to an array-like of per-row scores
        (e.g. ``lambda d: model.predict_proba(d)``).
    row:
        A single record (``pandas.Series``) following the canonical schema.
    field:
        Sequence column to perturb (``"peptide"``, ``"cdr3b"``, ...).
    cfg:
        Unused placeholder kept for interface symmetry / forward compatibility.
    neutral:
        Residue substituted at each position.
    relative:
        If True, normalise the per-residue drops by the base score so the result
        is a fractional importance; otherwise raw score drops are returned.

    Returns
    -------
    numpy.ndarray
        A ``[L]`` float array (``L == len(row[field])``) of importance values,
        one per residue. Empty sequences yield an empty array.

    Raises
    ------
    KeyError
        If ``field`` is not present in ``row``.
    """
    if field not in row.index:
        raise KeyError(f"field {field!r} not in record (have {list(row.index)})")

    seq = "" if pd.isna(row[field]) else str(row[field])
    length = len(seq)
    if length == 0:
        _log.debug("residue_attribution: empty %s field, returning empty array", field)
        return np.zeros(0, dtype=np.float64)

    base_row = row.to_frame().T.reset_index(drop=True)
    base_score = float(np.asarray(predict_fn(base_row)).reshape(-1)[0])

    # Build one mutated row per position and score them in a single batched call.
    mutated_rows = []
    for i in range(length):
        sub = neutral if seq[i] != neutral else _alt_residue(neutral)
        mutated = seq[:i] + sub + seq[i + 1 :]
        new_row = row.copy()
        new_row[field] = mutated
        mutated_rows.append(new_row)

    mutated_df = pd.DataFrame(mutated_rows).reset_index(drop=True)
    mutated_scores = np.asarray(predict_fn(mutated_df), dtype=np.float64).reshape(-1)

    attr = base_score - mutated_scores
    if relative and abs(base_score) > 1e-12:
        attr = attr / base_score

    attr = np.nan_to_num(attr, nan=0.0, posinf=0.0, neginf=0.0)
    _log.debug(
        "residue_attribution(%s): L=%d base=%.4f max=%.4f",
        field,
        length,
        base_score,
        float(np.max(np.abs(attr))) if attr.size else 0.0,
    )
    return attr.astype(np.float64)


def _alt_residue(neutral: str) -> str:
    """Pick a replacement residue when the native one equals ``neutral``.

    Parameters
    ----------
    neutral:
        The primary substitution residue (typically alanine).

    Returns
    -------
    str
        Glycine when ``neutral`` is alanine, else alanine; ensures the
        substitution actually changes the sequence at alanine positions.
    """
    return "G" if neutral.upper() == "A" else "A"


def attention_rollout(
    model: object,
    df: pd.DataFrame,
    *,
    field: str = "peptide",
    layer: int = -1,
) -> list[np.ndarray]:
    """Attention-rollout per-residue importance for a torch model (optional).

    This is a best-effort, lazily-imported path for models that expose a torch
    encoder with attention. It is **not** required for the baseline; prefer
    :func:`residue_attribution` for model-agnostic explanations.

    Parameters
    ----------
    model:
        A torch-backed model exposing an ``encoder`` / ``forward`` returning
        attention weights, or an ``attention_weights`` attribute after a forward
        pass.
    df:
        Records to explain.
    field:
        Sequence field whose residues the importance maps onto.
    layer:
        Which attention layer to roll out from (``-1`` = last).

    Returns
    -------
    list of numpy.ndarray
        One per-residue importance array per record.

    Raises
    ------
    ImportError
        If ``torch`` is not installed.
    NotImplementedError
        If the provided ``model`` does not expose attention weights.
    """
    try:
        import torch  # noqa: F401  # lazy: optional heavy dependency
    except ImportError as exc:  # pragma: no cover - torch is installed in CI
        raise ImportError(
            "attention_rollout requires 'torch'. Install it with "
            "`pip install 'tcr-cliff[torch]'` or use residue_attribution instead."
        ) from exc

    attn = getattr(model, "attention_weights", None)
    if attn is None:
        raise NotImplementedError(
            "model does not expose attention weights; use residue_attribution "
            "(model-agnostic occlusion) for this model"
        )
    raise NotImplementedError(
        "attention_rollout is only implemented for transformer-attention models; "
        "use residue_attribution for the baseline and MLP-based models"
    )
