# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""ECFP-free baseline: ESM (or fallback) embeddings + gradient-boosted trees.

Uses LightGBM when available (the spec's reference baseline) and transparently
falls back to scikit-learn's :class:`HistGradientBoostingClassifier` otherwise, so
the baseline trains and evaluates on a minimal install.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from tcr_cliff._logging import get_logger
from tcr_cliff.config import Config
from tcr_cliff.models.features import sequence_features

_log = get_logger("models.baseline")


def lightgbm_available() -> bool:
    """True if the optional LightGBM dependency can be imported."""
    try:
        import lightgbm  # noqa: F401

        return True
    except Exception:
        return False


class BaselineModel:
    """Gradient-boosted classifier over concatenated sequence embeddings."""

    kind = "baseline_lgbm"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.estimator = None
        self.feature_names: list[str] = []
        self.backend_used: str = ""

    # -- training -----------------------------------------------------------
    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame | None = None) -> BaselineModel:
        X, names = sequence_features(df_train, self.cfg)
        y = df_train["binder"].to_numpy().astype(int)
        self.feature_names = names
        params = self.cfg.model.baseline
        if params.prefer == "lightgbm" and lightgbm_available():
            import lightgbm as lgb

            self.estimator = lgb.LGBMClassifier(
                n_estimators=params.n_estimators,
                learning_rate=params.learning_rate,
                num_leaves=params.num_leaves,
                max_depth=params.max_depth,
                subsample=params.subsample,
                colsample_bytree=params.colsample_bytree,
                reg_lambda=params.reg_lambda,
                random_state=self.cfg.seed,
                n_jobs=-1,
                verbose=-1,
            )
            fit_kw = {}
            if df_val is not None and len(df_val):
                Xv, _ = sequence_features(df_val, self.cfg)
                fit_kw = {"eval_set": [(Xv, df_val["binder"].to_numpy().astype(int))]}
            self.estimator.fit(X, y, **fit_kw)
            self.backend_used = "lightgbm"
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier

            self.estimator = HistGradientBoostingClassifier(
                learning_rate=params.learning_rate,
                max_iter=params.n_estimators,
                max_leaf_nodes=params.num_leaves,
                l2_regularization=params.reg_lambda,
                random_state=self.cfg.seed,
            )
            self.estimator.fit(X, y)
            self.backend_used = "sklearn-histgbm"
        _log.info(
            "baseline trained on %d rows (%d feats) via %s",
            len(df_train),
            X.shape[1],
            self.backend_used,
        )
        return self

    # -- inference ----------------------------------------------------------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("model is not fitted")
        X, _ = sequence_features(df, self.cfg)
        return self.estimator.predict_proba(X)[:, 1].astype(np.float32)

    # -- persistence --------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "meta.json").write_text(
            json.dumps(
                {
                    "kind": self.kind,
                    "backend_used": self.backend_used,
                    "feature_names": self.feature_names,
                }
            )
        )
        joblib.dump(self.estimator, path / "estimator.joblib")
        self.cfg.to_yaml(path / "config.yaml")
        _log.info("saved baseline model to %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> BaselineModel:
        from tcr_cliff.config import load_config

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        cfg = load_config(path / "config.yaml")
        obj = cls(cfg)
        obj.estimator = joblib.load(path / "estimator.joblib")
        obj.feature_names = meta.get("feature_names", [])
        obj.backend_used = meta.get("backend_used", "")
        return obj
