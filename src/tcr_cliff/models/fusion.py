# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Structure-fusion model: cliff-aware sequence encoder + structural-confidence feats.

:class:`FusionModel` (registered under ``kind="fusion"``) extends
:class:`~tcr_cliff.models.head.CliffAwareModel`. The sequence half is unchanged —
the same contrastively-pretrained cliff-aware encoder — but the binding head's
input is ``concat(sequence_rep, structure_features(df, cfg))``, fusing AlphaFold/
ESMFold confidence features (pTM/ipTM, pLDDT, interface PAE, ...) onto the
sequence representation.

The model satisfies the ``Model`` protocol and ``save``/``load`` mirror the
cliff-aware model with ``meta.json{"kind": "fusion"}``. ``torch`` is lazy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tcr_cliff._logging import get_logger
from tcr_cliff.config import Config
from tcr_cliff.models.contrastive import contrastive_pretrain, make_encoder
from tcr_cliff.models.dataset import build_pair_index
from tcr_cliff.models.features import structure_features
from tcr_cliff.models.head import BindingHead, CliffAwareModel, train_binding_head
from tcr_cliff.utils.seed import seed_everything

_log = get_logger("models.fusion")


def _import_torch() -> Any:
    """Import torch lazily, raising an actionable error if it is missing."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "PyTorch is required for the cliff-aware/fusion models. "
            "Install it with: pip install 'tcr-cliff[torch]' (or `pip install torch`)."
        ) from exc
    return torch


class FusionModel(CliffAwareModel):
    """Cliff-aware sequence model fused with structural-confidence features."""

    kind = "fusion"

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.struct_names: list[str] = []

    # -- features -----------------------------------------------------------
    def _structure(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Compute per-pair structural-confidence features for ``df``."""
        return structure_features(df, self.cfg)

    # -- training -----------------------------------------------------------
    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame | None = None) -> FusionModel:
        """Fit the fusion model: contrastive encoder + structure-fused head.

        Parameters
        ----------
        df_train:
            Training pairs table (must contain a ``binder`` column).
        df_val:
            Optional validation table (kept for interface symmetry).

        Returns
        -------
        FusionModel
            ``self``, fitted.
        """
        seed_everything(self.cfg.train.seed)
        X_seq, names = self._features(df_train)
        X_struct, struct_names = self._structure(df_train)
        y = df_train["binder"].to_numpy().astype(np.float32)

        self.feature_names = names
        self.struct_names = struct_names
        self.in_dim = int(X_seq.shape[1])
        self.extra_dim = int(X_struct.shape[1])

        pairs = build_pair_index(df_train, self.cfg)
        encoder = make_encoder(self.in_dim, self.cfg)
        encoder = contrastive_pretrain(encoder, X_seq, pairs, self.cfg)

        head = BindingHead(
            self.in_dim,
            self.cfg,
            extra_dim=self.extra_dim,
            freeze_encoder=self.cfg.model.cliff_aware.freeze_encoder_after_pretrain,
            encoder=encoder,
        )
        self.head = train_binding_head(head, X_seq, y, self.cfg, X_extra=X_struct)
        _log.info(
            "fusion trained on %d rows (seq_dim=%d, struct_dim=%d, %d pairs)",
            len(df_train),
            self.in_dim,
            self.extra_dim,
            len(pairs),
        )
        return self

    # -- inference ----------------------------------------------------------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return positive-class binding probabilities in ``[0, 1]`` for ``df``."""
        if self.head is None:
            raise RuntimeError("model is not fitted")
        torch = _import_torch()
        X_seq, _ = self._features(df)
        X_struct, _ = self._structure(df)
        device = next(self.head.parameters()).device
        x_seq = torch.as_tensor(np.asarray(X_seq, dtype=np.float32), device=device)
        x_struct = torch.as_tensor(np.asarray(X_struct, dtype=np.float32), device=device)
        self.head.eval()
        with torch.no_grad():
            logits = self.head(x_seq, x_struct)
            probs = torch.sigmoid(logits)
        return probs.detach().cpu().numpy().astype(np.float32)

    # -- persistence --------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Persist the fusion model (``meta.json{"kind": "fusion"}`` + weights)."""
        torch = _import_torch()
        if self.head is None:
            raise RuntimeError("cannot save an unfitted model")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "meta.json").write_text(
            json.dumps(
                {
                    "kind": self.kind,
                    "in_dim": self.in_dim,
                    "extra_dim": self.extra_dim,
                    "feature_names": self.feature_names,
                    "struct_names": self.struct_names,
                }
            )
        )
        torch.save(self.head.state_dict(), path / "weights.pt")
        self.cfg.to_yaml(path / "config.yaml")
        _log.info("saved %s model to %s", self.kind, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> FusionModel:
        """Load a fusion model previously written by :meth:`save`."""
        from tcr_cliff.config import load_config

        torch = _import_torch()
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        cfg = load_config(path / "config.yaml")
        obj = cls(cfg)
        obj.in_dim = int(meta["in_dim"])
        obj.extra_dim = int(meta.get("extra_dim", cfg.model.fusion.structure_feature_dim))
        obj.feature_names = meta.get("feature_names", [])
        obj.struct_names = meta.get("struct_names", [])
        head = BindingHead(
            obj.in_dim,
            cfg,
            extra_dim=obj.extra_dim,
            freeze_encoder=cfg.model.cliff_aware.freeze_encoder_after_pretrain,
        )
        state = torch.load(path / "weights.pt", map_location="cpu")
        head.load_state_dict(state)
        head.eval()
        obj.head = head
        return obj
