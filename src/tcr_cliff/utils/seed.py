# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Deterministic seeding and stable hashing helpers."""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np

from tcr_cliff._logging import get_logger

_log = get_logger("utils.seed")


def seed_everything(seed: int = 0, *, deterministic_torch: bool = True) -> int:
    """Seed Python, NumPy, and (if available) PyTorch RNGs.

    Parameters
    ----------
    seed:
        The integer seed.
    deterministic_torch:
        If ``True`` and torch is importable, also set deterministic cudnn flags.

    Returns
    -------
    int
        The seed that was applied (echoed for logging/config provenance).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is an optional dependency.
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:  # pragma: no cover - torch not installed
        _log.debug("torch not available; skipping torch seeding")
    _log.debug("seeded all RNGs with %d", seed)
    return seed


def stable_hash(text: str, *, length: int = 16) -> str:
    """Return a deterministic hex digest of ``text``.

    Unlike :func:`hash`, this is stable across interpreter runs (it does not
    depend on ``PYTHONHASHSEED``), which makes it safe for cache keys.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:length]
