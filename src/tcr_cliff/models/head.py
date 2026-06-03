# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Cliff-aware supervised model: contrastive encoder + binding-prediction head.

:class:`CliffAwareModel` is the torch-backed model registered under
``kind="cliff_aware"``. Training proceeds in two stages:

1. **Contrastive pretraining** of a sequence encoder over mined neighbour pairs
   (pull smooth pairs together, push cliff pairs apart) — see
   :mod:`tcr_cliff.models.contrastive`.
2. **Supervised fine-tuning** of a :class:`BindingHead` (the encoder, optionally
   frozen, plus a linear classifier) with binary cross-entropy on the ``binder``
   label.

The model satisfies the ``Model`` protocol: ``predict_proba(df) -> np.ndarray`` in
``[0, 1]``, ``save(dir) -> Path`` writing ``meta.json``/``weights.pt``/
``config.yaml``, and classmethod ``load(dir)``. ``torch`` is imported lazily.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from tcr_cliff._logging import get_logger
from tcr_cliff.config import Config
from tcr_cliff.models.contrastive import contrastive_pretrain, make_encoder
from tcr_cliff.models.dataset import build_pair_index
from tcr_cliff.models.features import sequence_features
from tcr_cliff.utils.seed import seed_everything

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

_log = get_logger("models.head")


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


def _build_binding_head_class() -> type:
    """Build the ``BindingHead`` class bound to ``torch.nn`` lazily."""
    torch = _import_torch()
    nn = torch.nn

    class _BindingHead(nn.Module):
        """Cliff-aware encoder + linear classifier producing a binding logit.

        The encoder maps sequence features to a projection; an optional
        ``extra_dim`` (structure features in the fusion model) is concatenated onto
        that projection before a linear classifier produces a single logit. When
        ``freeze_encoder`` is set the encoder parameters are detached so only the
        classifier (and any extra projection) is trained.

        Parameters
        ----------
        in_dim:
            Sequence-feature input dimension.
        cfg:
            Run configuration; ``cfg.model.cliff_aware`` drives encoder geometry.
        extra_dim:
            Extra (e.g. structure) feature dimension concatenated post-encoder.
        freeze_encoder:
            If True, the encoder is run under ``no_grad`` during the forward pass.
        encoder:
            Optionally reuse a pretrained encoder instead of building a fresh one.
        """

        def __init__(
            self,
            in_dim: int,
            cfg: Config,
            extra_dim: int = 0,
            freeze_encoder: bool = False,
            encoder: Any | None = None,
        ) -> None:
            super().__init__()
            self.in_dim = int(in_dim)
            self.extra_dim = int(extra_dim)
            self.freeze_encoder = bool(freeze_encoder)
            if encoder is None:
                encoder = make_encoder(in_dim, cfg)
            self.encoder = encoder
            head_in = self.encoder.proj_dim + self.extra_dim
            self.classifier = nn.Linear(head_in, 1)

        def encode(self, x_seq: torch.Tensor) -> torch.Tensor:
            """Run the encoder, respecting the ``freeze_encoder`` flag."""
            if self.freeze_encoder:
                with torch.no_grad():
                    return self.encoder(x_seq)
            return self.encoder(x_seq)

        def forward(self, x_seq: torch.Tensor, x_extra: torch.Tensor | None = None) -> torch.Tensor:
            """Return binding logits ``[B]`` for sequence (+ optional extra) feats."""
            z = self.encode(x_seq)
            if self.extra_dim and x_extra is not None:
                z = torch.cat([z, x_extra], dim=-1)
            return self.classifier(z).squeeze(-1)

    return _BindingHead


_HEAD_CLS: type | None = None


def get_binding_head_class() -> type:
    """Return the concrete ``BindingHead`` ``nn.Module`` class (lazy torch)."""
    global _HEAD_CLS
    if _HEAD_CLS is None:
        _HEAD_CLS = _build_binding_head_class()
    return _HEAD_CLS


def BindingHead(*args: Any, **kwargs: Any) -> Any:
    """Instantiate the binding head (lazily binds to ``torch.nn.Module``)."""
    return get_binding_head_class()(*args, **kwargs)


def _resolve_device(device: str) -> Any:
    """Resolve a torch device, falling back to CPU when CUDA is unavailable."""
    torch = _import_torch()
    if device.startswith("cuda") and not torch.cuda.is_available():
        _log.warning("CUDA requested but not available; using CPU")
        return torch.device("cpu")
    return torch.device(device)


def train_binding_head(
    head: Any,
    X_seq: np.ndarray,
    y: np.ndarray,
    cfg: Config,
    *,
    X_extra: np.ndarray | None = None,
) -> Any:
    """Fine-tune a :class:`BindingHead` with binary cross-entropy.

    Parameters
    ----------
    head:
        The binding head module to train.
    X_seq:
        Sequence-feature matrix ``[N, D]``.
    y:
        Binary labels ``[N]`` in ``{0, 1}``.
    cfg:
        Run configuration; ``cfg.model.cliff_aware`` and ``cfg.train`` are read.
    X_extra:
        Optional extra feature matrix ``[N, extra_dim]`` (fusion structure feats).

    Returns
    -------
    BindingHead
        The trained head (also trained in place).
    """
    torch = _import_torch()
    seed_everything(cfg.train.seed)
    mcfg = cfg.model.cliff_aware
    device = _resolve_device(cfg.train.device)
    head = head.to(device)

    x_seq_t = torch.as_tensor(np.asarray(X_seq, dtype=np.float32), device=device)
    y_t = torch.as_tensor(np.asarray(y, dtype=np.float32), device=device)
    x_extra_t = (
        torch.as_tensor(np.asarray(X_extra, dtype=np.float32), device=device)
        if X_extra is not None
        else None
    )

    n = x_seq_t.shape[0]
    batch_size = max(1, min(mcfg.batch_size, n))
    optim = torch.optim.Adam(
        (p for p in head.parameters() if p.requires_grad),
        lr=mcfg.lr,
        weight_decay=mcfg.weight_decay,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    log_every = max(1, int(cfg.train.log_every))

    head.train()
    for epoch in range(max(0, mcfg.head_epochs)):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = x_seq_t[idx]
            xe = x_extra_t[idx] if x_extra_t is not None else None
            yb = y_t[idx]
            optim.zero_grad()
            logits = head(xb, xe)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
        if n_batches and (epoch % log_every == 0 or epoch == mcfg.head_epochs - 1):
            _log.info(
                "head epoch %d/%d bce=%.4f", epoch + 1, mcfg.head_epochs, epoch_loss / n_batches
            )
    head.eval()
    return head


class CliffAwareModel:
    """Contrastive cliff-aware encoder + supervised binding head.

    Conforms to the ``Model`` protocol used across tcr-cliff. The sequence encoder
    is pretrained contrastively on mined neighbour pairs before a linear head is
    fine-tuned with BCE on the binding label.
    """

    kind = "cliff_aware"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.head: Any | None = None
        self.in_dim: int = 0
        self.extra_dim: int = 0
        self.feature_names: list[str] = []

    # -- training -----------------------------------------------------------
    def _features(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Compute the sequence feature matrix and names for ``df``."""
        return sequence_features(df, self.cfg)

    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame | None = None) -> CliffAwareModel:
        """Fit the cliff-aware model on a pairs table.

        Computes sequence features, mines neighbour pairs, runs contrastive
        pretraining of the encoder, then fine-tunes the binding head.

        Parameters
        ----------
        df_train:
            Training pairs table (must contain a ``binder`` column).
        df_val:
            Optional validation table (currently unused for early stopping; kept
            for interface symmetry with the baseline).

        Returns
        -------
        CliffAwareModel
            ``self``, fitted.
        """
        seed_everything(self.cfg.train.seed)
        X, names = self._features(df_train)
        y = df_train["binder"].to_numpy().astype(np.float32)
        self.feature_names = names
        self.in_dim = int(X.shape[1])

        pairs = build_pair_index(df_train, self.cfg)
        encoder = make_encoder(self.in_dim, self.cfg)
        encoder = contrastive_pretrain(encoder, X, pairs, self.cfg)

        head = self._build_head(encoder)
        self.head = train_binding_head(head, X, y, self.cfg)
        _log.info(
            "cliff_aware trained on %d rows (in_dim=%d, %d pairs)",
            len(df_train),
            self.in_dim,
            len(pairs),
        )
        return self

    def _build_head(self, encoder: Any) -> Any:
        """Build the binding head reusing the (pretrained) ``encoder``."""
        return BindingHead(
            self.in_dim,
            self.cfg,
            extra_dim=0,
            freeze_encoder=self.cfg.model.cliff_aware.freeze_encoder_after_pretrain,
            encoder=encoder,
        )

    # -- inference ----------------------------------------------------------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return positive-class binding probabilities in ``[0, 1]`` for ``df``."""
        if self.head is None:
            raise RuntimeError("model is not fitted")
        torch = _import_torch()
        X, _ = self._features(df)
        device = next(self.head.parameters()).device
        x = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
        self.head.eval()
        with torch.no_grad():
            logits = self.head(x, None)
            probs = torch.sigmoid(logits)
        return probs.detach().cpu().numpy().astype(np.float32)

    # -- persistence --------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Persist the model as a directory (``meta.json``/``weights.pt``/config)."""
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
                }
            )
        )
        torch.save(self.head.state_dict(), path / "weights.pt")
        self.cfg.to_yaml(path / "config.yaml")
        _log.info("saved %s model to %s", self.kind, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> CliffAwareModel:
        """Load a model previously written by :meth:`save`."""
        from tcr_cliff.config import load_config

        torch = _import_torch()
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        cfg = load_config(path / "config.yaml")
        obj = cls(cfg)
        obj.in_dim = int(meta["in_dim"])
        obj.extra_dim = int(meta.get("extra_dim", 0))
        obj.feature_names = meta.get("feature_names", [])
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
