# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""SHAP-based explanation of the gradient-boosted baseline.

The baseline operates on concatenated per-field sequence embeddings whose column
names look like ``"<field>_<j>"`` (see
:func:`tcr_cliff.embeddings.build_feature_matrix`). Raw SHAP values over those
hundreds of embedding dimensions are hard to read, so :func:`explain_baseline`
sums the absolute SHAP contribution within each field's columns to produce a
compact, interpretable per-field importance block (``cdr3b``, ``peptide``,
``mhc_pseudo``).
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from tcr_cliff._logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tcr_cliff.models.baseline import BaselineModel

_log = get_logger("interpret.shap_baseline")


def _field_of(name: str) -> str:
    """Return the field prefix of a ``"<field>_<j>"`` feature name.

    Parameters
    ----------
    name:
        Feature/column name produced by ``build_feature_matrix``.

    Returns
    -------
    str
        The portion before the final underscore (the originating sequence field),
        or the whole name if it contains no underscore.
    """
    head, sep, _tail = name.rpartition("_")
    return head if sep else name


def _shap_values_2d(explainer_output: Any, n_features: int) -> np.ndarray:
    """Coerce a SHAP explainer output into a ``[n_samples, n_features]`` array.

    SHAP returns several shapes depending on the explainer and model family
    (a list of per-class arrays for binary tree models, an ``Explanation``
    object, or a plain array). This normalises all of them to the contribution
    of the positive class.

    Parameters
    ----------
    explainer_output:
        Raw return value of a ``shap`` explainer call.
    n_features:
        Expected number of feature columns, used to disambiguate axes.

    Returns
    -------
    numpy.ndarray
        A 2-D ``[n_samples, n_features]`` array of SHAP values.
    """
    vals = getattr(explainer_output, "values", explainer_output)
    if isinstance(vals, list):  # one array per class (binary -> take positive)
        vals = vals[-1]
    vals = np.asarray(vals, dtype=np.float64)
    if vals.ndim == 1:
        vals = vals.reshape(1, -1)
    if vals.ndim == 3:
        # [n_samples, n_features, n_classes] -> positive class
        vals = vals[:, :, -1]
    if vals.ndim == 2 and vals.shape[1] != n_features and vals.shape[0] == n_features:
        vals = vals.T
    return vals


def explain_baseline(
    model: BaselineModel,
    df: pd.DataFrame,
    *,
    max_display: int = 20,
    background: pd.DataFrame | None = None,
) -> dict:
    """Explain a fitted baseline with SHAP, aggregated to per-field blocks.

    SHAP values are computed over the embedding features and then collapsed back
    onto the originating sequence fields by summing the mean absolute SHAP value
    within each field's column block. This answers the practical question "which
    *sequence* (CDR3b vs peptide vs MHC) drives this binder prediction?" rather
    than reporting opaque embedding dimensions.

    Parameters
    ----------
    model:
        A fitted :class:`tcr_cliff.models.baseline.BaselineModel`. Must expose
        ``estimator`` and ``feature_names``.
    df:
        Records to explain (canonical pairs schema). The same feature pipeline as
        training is applied via the model's config.
    max_display:
        Number of top individual embedding features to retain in the
        ``top_features`` summary.
    background:
        Optional background sample used by non-tree explainers. When omitted, the
        explained data itself is used as background.

    Returns
    -------
    dict
        A mapping with keys:

        ``field_importance``
            ``{field: float}`` summed mean-|SHAP| per sequence field.
        ``field_importance_normalised``
            The same, normalised to sum to 1 (empty if all zero).
        ``top_features``
            List of ``(feature_name, mean_abs_shap)`` for the largest
            ``max_display`` features.
        ``mean_abs_shap``
            ``np.ndarray`` of per-feature mean absolute SHAP values.
        ``feature_names``
            The feature name list the values align to.
        ``explainer``
            String tag of the explainer used (``"tree"``/``"generic"``/
            ``"permutation"``).
        ``n_samples``
            Number of records explained.

    Raises
    ------
    RuntimeError
        If the model has not been fitted.
    ImportError
        If the optional ``shap`` dependency is unavailable for tree/generic paths.
    """
    if getattr(model, "estimator", None) is None:
        raise RuntimeError("model is not fitted; call .fit(...) before explaining")

    from tcr_cliff.models.features import sequence_features

    X, names = sequence_features(df, model.cfg)
    feature_names = list(model.feature_names) or names
    n_features = X.shape[1]

    mean_abs, explainer_tag = _compute_mean_abs_shap(
        model, X, background=background, n_features=n_features
    )

    field_importance: dict[str, float] = defaultdict(float)
    for name, val in zip(feature_names, mean_abs):
        field_importance[_field_of(name)] += float(val)
    field_importance = dict(field_importance)

    total = float(sum(field_importance.values()))
    field_norm = {k: v / total for k, v in field_importance.items()} if total > 0 else {}

    order = np.argsort(mean_abs)[::-1][:max_display]
    top_features = [(feature_names[i], float(mean_abs[i])) for i in order]

    _log.info(
        "explain_baseline: %d records, explainer=%s, fields=%s",
        len(df),
        explainer_tag,
        sorted(field_importance),
    )
    return {
        "field_importance": field_importance,
        "field_importance_normalised": field_norm,
        "top_features": top_features,
        "mean_abs_shap": mean_abs,
        "feature_names": feature_names,
        "explainer": explainer_tag,
        "n_samples": len(df),
    }


def _compute_mean_abs_shap(
    model: BaselineModel,
    X: np.ndarray,
    *,
    background: pd.DataFrame | None,
    n_features: int,
) -> tuple[np.ndarray, str]:
    """Compute per-feature mean absolute SHAP values for a fitted baseline.

    Uses :class:`shap.TreeExplainer` for LightGBM-style tree models, falls back to
    the generic :class:`shap.Explainer`, and finally to a model-agnostic
    permutation importance if SHAP cannot handle the estimator.

    Parameters
    ----------
    model:
        Fitted baseline model wrapping a tree estimator.
    X:
        Feature matrix ``[N, n_features]`` to explain.
    background:
        Optional background records for non-tree explainers.
    n_features:
        Number of feature columns (for output shaping).

    Returns
    -------
    (mean_abs, tag):
        ``mean_abs`` is a ``[n_features]`` array of mean absolute SHAP values;
        ``tag`` identifies the explainer that produced them.
    """
    estimator = model.estimator
    try:
        import shap  # lazy: optional heavy dependency
    except ImportError as exc:  # pragma: no cover - shap is installed in CI
        raise ImportError(
            "explain_baseline requires 'shap'. Install it with "
            "`pip install 'tcr-cliff[interpret]'` or `pip install shap`."
        ) from exc

    bg = None
    if background is not None and len(background):
        from tcr_cliff.models.features import sequence_features

        bg, _ = sequence_features(background, model.cfg)

    try:
        explainer = shap.TreeExplainer(estimator)
        raw = explainer.shap_values(X)
        vals = _shap_values_2d(raw, n_features)
        return np.abs(vals).mean(axis=0), "tree"
    except Exception as exc:
        _log.debug("TreeExplainer failed (%s); trying generic Explainer", exc)

    try:
        data = bg if bg is not None else X
        explainer = shap.Explainer(estimator.predict_proba, data)
        raw = explainer(X)
        vals = _shap_values_2d(raw, n_features)
        return np.abs(vals).mean(axis=0), "generic"
    except Exception as exc:
        _log.debug("generic Explainer failed (%s); using permutation", exc)

    return _permutation_importance(estimator, X), "permutation"


def _permutation_importance(estimator: Any, X: np.ndarray) -> np.ndarray:
    """Model-agnostic permutation importance fallback when SHAP cannot run.

    Each feature column is shuffled and the mean absolute change in predicted
    positive-class probability is recorded as that feature's importance.

    Parameters
    ----------
    estimator:
        A fitted estimator exposing ``predict_proba``.
    X:
        Feature matrix ``[N, n_features]``.

    Returns
    -------
    numpy.ndarray
        A ``[n_features]`` non-negative importance vector.
    """
    rng = np.random.default_rng(0)
    base = estimator.predict_proba(X)[:, 1]
    imp = np.zeros(X.shape[1], dtype=np.float64)
    for j in range(X.shape[1]):
        Xp = X.copy()
        rng.shuffle(Xp[:, j])
        perturbed = estimator.predict_proba(Xp)[:, 1]
        imp[j] = float(np.abs(perturbed - base).mean())
    return imp
