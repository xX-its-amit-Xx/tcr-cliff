# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Cliff-aware contrastive pretraining of a sequence encoder.

The encoder is a small MLP that maps concatenated per-field sequence embeddings
into an L2-normalised projection space. It is pretrained with a contrastive
objective over mined neighbour pairs:

* **smooth** (non-cliff) neighbour pairs are *pulled together* — near-identical
  sequences with the same binding outcome should sit close;
* **cliff** pairs are *pushed apart* (up to a margin) — single-residue neighbours
  whose binding outcome flips must be separable in the representation.

The push term is up-weighted by ``cfg.model.cliff_aware.cliff_weight`` so the
geometry is dominated by getting the hard, label-flipping neighbours right. The
``contrastive_temperature`` rescales cosine distances and ``contrastive_margin``
sets how far cliff pairs are driven apart.

``torch`` is imported lazily; nothing here is evaluated at module import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tcr_cliff._logging import get_logger
from tcr_cliff.cliffs.detect import NeighborPair
from tcr_cliff.config import Config
from tcr_cliff.utils.seed import seed_everything

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

_log = get_logger("models.contrastive")


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


def make_encoder(in_dim: int, cfg: Config) -> Any:
    """Construct a :class:`CliffAwareEncoder` from a config.

    Parameters
    ----------
    in_dim:
        Dimensionality of the concatenated sequence features.
    cfg:
        Run configuration; ``cfg.model.cliff_aware`` supplies hidden/proj dims.

    Returns
    -------
    CliffAwareEncoder
        An untrained encoder module.
    """
    mcfg = cfg.model.cliff_aware
    return CliffAwareEncoder(
        in_dim=in_dim,
        hidden_dim=mcfg.hidden_dim,
        proj_dim=mcfg.proj_dim,
        dropout=mcfg.dropout,
    )


def _build_encoder_class() -> type:
    """Build the ``CliffAwareEncoder`` class bound to ``torch.nn`` lazily."""
    torch = _import_torch()
    nn = torch.nn

    class _CliffAwareEncoder(nn.Module):
        """MLP encoder mapping sequence features to an L2-normalised projection.

        Architecture: ``in_dim -> hidden_dim -> proj_dim`` with ReLU and dropout
        between the two linear layers; the output is L2-normalised so distances in
        projection space are cosine distances.

        Parameters
        ----------
        in_dim:
            Input feature dimension.
        hidden_dim:
            Hidden layer width.
        proj_dim:
            Output projection dimension.
        dropout:
            Dropout probability applied after the hidden activation.
        """

        def __init__(
            self,
            in_dim: int,
            hidden_dim: int = 256,
            proj_dim: int = 128,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.in_dim = int(in_dim)
            self.hidden_dim = int(hidden_dim)
            self.proj_dim = int(proj_dim)
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, proj_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Encode ``x`` ``[B, in_dim]`` to an L2-normalised ``[B, proj_dim]``."""
            z = self.net(x)
            return torch.nn.functional.normalize(z, p=2.0, dim=-1)

    return _CliffAwareEncoder


# ``CliffAwareEncoder`` is materialised on first access so that importing this
# module never imports torch. ``make_encoder`` and the head construct it directly.
_ENCODER_CLS: type | None = None


def CliffAwareEncoder(*args: Any, **kwargs: Any) -> Any:
    """Instantiate the cliff-aware encoder (lazily binds to ``torch.nn.Module``).

    Accepts the same arguments as the underlying module:
    ``CliffAwareEncoder(in_dim, hidden_dim=256, proj_dim=128, dropout=0.1)``.
    """
    global _ENCODER_CLS
    if _ENCODER_CLS is None:
        _ENCODER_CLS = _build_encoder_class()
    return _ENCODER_CLS(*args, **kwargs)


def get_encoder_class() -> type:
    """Return the concrete ``nn.Module`` encoder class (importing torch lazily)."""
    global _ENCODER_CLS
    if _ENCODER_CLS is None:
        _ENCODER_CLS = _build_encoder_class()
    return _ENCODER_CLS


def cliff_contrastive_loss(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    is_cliff: torch.Tensor,
    *,
    margin: float,
    temperature: float,
    cliff_weight: float,
) -> torch.Tensor:
    """Cliff-aware contrastive loss over a batch of neighbour pairs.

    Uses the (squared) Euclidean distance between L2-normalised projections, which
    is monotonic in cosine distance. Smooth pairs minimise distance; cliff pairs
    minimise ``relu(margin - distance)`` so they are driven at least ``margin``
    apart. ``temperature`` rescales the distance and ``cliff_weight`` up-weights
    the cliff (push) term.

    Parameters
    ----------
    z_i, z_j:
        L2-normalised projections ``[B, P]`` for each side of the pair.
    is_cliff:
        Float ``[B]`` mask, ``1.0`` for cliff pairs and ``0.0`` for smooth.
    margin:
        Minimum target distance for cliff pairs.
    temperature:
        Positive scale applied to distances (smaller = sharper).
    cliff_weight:
        Multiplier on the cliff push term relative to the smooth pull term.

    Returns
    -------
    torch.Tensor
        Scalar mean loss over the batch.
    """
    torch = _import_torch()
    temp = max(float(temperature), 1e-6)
    # Squared L2 distance between unit vectors lies in [0, 4].
    dist = ((z_i - z_j) ** 2).sum(dim=-1) / temp
    smooth_term = (1.0 - is_cliff) * dist
    push = torch.clamp(margin - dist, min=0.0)
    cliff_term = is_cliff * cliff_weight * push
    return (smooth_term + cliff_term).mean()


def contrastive_pretrain(
    encoder: Any,
    X: np.ndarray,
    pairs: list[NeighborPair],
    cfg: Config,
) -> Any:
    """Pretrain ``encoder`` with the cliff-aware contrastive objective.

    Pulls smooth neighbour pairs together and pushes cliff pairs apart in the
    encoder's projection space. RNGs are seeded via
    :func:`seed_everything` for reproducibility. No-ops gracefully (returns the
    encoder unchanged) when there are no usable pairs.

    Parameters
    ----------
    encoder:
        A :class:`CliffAwareEncoder` (``nn.Module``).
    X:
        Feature matrix ``[N, D]`` aligned to the records the pairs index into.
    pairs:
        Mined neighbour pairs (cliffs + smooth).
    cfg:
        Run configuration; ``cfg.model.cliff_aware`` and ``cfg.train`` are read.

    Returns
    -------
    CliffAwareEncoder
        The pretrained encoder (trained in place, also returned).
    """
    from tcr_cliff.models.dataset import ContrastivePairs, make_pair_loader

    torch = _import_torch()
    seed_everything(cfg.train.seed)

    mcfg = cfg.model.cliff_aware
    if not pairs:
        _log.warning("no neighbour pairs available; skipping contrastive pretraining")
        return encoder
    if mcfg.contrastive_epochs <= 0:
        _log.info("contrastive_epochs<=0; skipping contrastive pretraining")
        return encoder

    device = _resolve_device(cfg.train.device)
    encoder = encoder.to(device)
    encoder.train()

    dataset = ContrastivePairs(np.asarray(X, dtype=np.float32), pairs)
    loader = make_pair_loader(dataset, batch_size=mcfg.batch_size, shuffle=True)
    optim = torch.optim.Adam(encoder.parameters(), lr=mcfg.lr, weight_decay=mcfg.weight_decay)

    log_every = max(1, int(cfg.train.log_every))
    for epoch in range(mcfg.contrastive_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x_i, x_j, is_cliff in loader:
            x_i = x_i.to(device)
            x_j = x_j.to(device)
            is_cliff = is_cliff.to(device)
            optim.zero_grad()
            z_i = encoder(x_i)
            z_j = encoder(x_j)
            loss = cliff_contrastive_loss(
                z_i,
                z_j,
                is_cliff,
                margin=mcfg.contrastive_margin,
                temperature=mcfg.contrastive_temperature,
                cliff_weight=mcfg.cliff_weight,
            )
            loss.backward()
            optim.step()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
        if n_batches and (epoch % log_every == 0 or epoch == mcfg.contrastive_epochs - 1):
            _log.info(
                "contrastive epoch %d/%d loss=%.4f",
                epoch + 1,
                mcfg.contrastive_epochs,
                epoch_loss / n_batches,
            )
    encoder.eval()
    return encoder


def _resolve_device(device: str) -> Any:
    """Resolve a torch device, falling back to CPU when CUDA is unavailable."""
    torch = _import_torch()
    if device.startswith("cuda") and not torch.cuda.is_available():
        _log.warning("CUDA requested but not available; using CPU")
        return torch.device("cpu")
    return torch.device(device)
