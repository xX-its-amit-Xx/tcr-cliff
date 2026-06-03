# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Fast tests for the torch-backed cliff-aware and fusion models.

These are skipped automatically when torch is unavailable. They are tuned to run
in a few seconds with tiny dimensions and very few epochs, exercising the full
fit / predict_proba / save / load round-trip for both ``cliff_aware`` and
``fusion`` models through the unified registry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from tcr_cliff.config import Config
from tcr_cliff.data.schema import validate_pairs
from tcr_cliff.models.dataset import build_pair_index
from tcr_cliff.models.registry import load_model, predict_scores, train_model

# --- fixtures --------------------------------------------------------------
_CDR3S = [
    "CASSLAPGATNEKLFF",
    "CASSPGTGNTEAFF",
    "CASSLGQAYEQYF",
    "CASSIRSSYEQYF",
]
_PEPTIDE_FAMILIES = [
    ("GILGFVFTL", "GILGFVFTV", "GILGFVFTA"),
    ("NLVPMVATV", "NLVPMVATI", "NLVPMVATL"),
    ("GLCTLVAML", "GLCTLVAMV", "GLCTLVAMA"),
]


def _toy_frame(n: int = 40) -> pd.DataFrame:
    """Build a small toy pairs table with single-residue peptide neighbours.

    The peptide families share a fixed CDR3/MHC context so that neighbour-pair
    mining produces both cliff (label-flip) and smooth pairs.
    """
    rng = np.random.default_rng(0)
    rows = []
    i = 0
    while len(rows) < n:
        fam = _PEPTIDE_FAMILIES[i % len(_PEPTIDE_FAMILIES)]
        cdr3 = _CDR3S[i % len(_CDR3S)]
        for k, pep in enumerate(fam):
            rows.append(
                {
                    "pair_id": f"p{len(rows)}",
                    "cdr3b": cdr3,
                    "peptide": pep,
                    "mhc": "HLA-A*02:01",
                    "mhc_pseudo": "YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY",
                    "binder": int((k + (i % 2)) % 2),  # flips within a family -> cliffs
                    "affinity": float(rng.uniform(0, 1)),
                    "source": "toy",
                    "split": "",
                }
            )
            if len(rows) >= n:
                break
        i += 1
    return validate_pairs(pd.DataFrame(rows))


def _fast_config(kind: str) -> Config:
    """A tiny, fast config for the requested model kind."""
    return Config.model_validate(
        {
            "seed": 0,
            "embedding": {"backend": "fallback", "fallback_dim": 64, "kmer_k": 2},
            "cliff": {"vary": "peptide", "max_edits": 1, "min_len": 5},
            "structure": {"enabled": True, "use_stub": True},
            "model": {
                "kind": kind,
                "cliff_aware": {
                    "proj_dim": 16,
                    "hidden_dim": 32,
                    "contrastive_epochs": 2,
                    "head_epochs": 3,
                    "batch_size": 16,
                },
                "fusion": {"hidden_dim": 32},
            },
            "train": {"seed": 0, "device": "cpu", "log_every": 1},
        }
    )


# --- tests -----------------------------------------------------------------
def test_pair_index_has_cliffs() -> None:
    """The toy frame yields neighbour pairs including at least one cliff."""
    cfg = _fast_config("cliff_aware")
    df = _toy_frame(40)
    pairs = build_pair_index(df, cfg)
    assert len(pairs) > 0
    assert any(p.is_cliff for p in pairs)
    assert any(not p.is_cliff for p in pairs)


def test_cliff_aware_fit_predict_save_load(tmp_path: Path) -> None:
    """cliff_aware: fit, probabilities in [0,1], and save/load round-trip."""
    cfg = _fast_config("cliff_aware")
    df = _toy_frame(40)

    model = train_model(cfg, df)
    assert model.kind == "cliff_aware"

    proba = model.predict_proba(df)
    assert proba.shape == (len(df),)
    assert np.all(np.isfinite(proba))
    assert proba.min() >= 0.0 and proba.max() <= 1.0

    out = model.save(tmp_path / "cliff_aware")
    assert (out / "meta.json").exists()
    assert (out / "weights.pt").exists()
    assert (out / "config.yaml").exists()

    reloaded = load_model(out)
    assert reloaded.kind == "cliff_aware"
    proba2 = reloaded.predict_proba(df)
    np.testing.assert_allclose(proba, proba2, rtol=1e-4, atol=1e-5)


def test_fusion_fit_predict_save_load(tmp_path: Path) -> None:
    """fusion: structure-fused fit, probabilities in [0,1], save/load round-trip."""
    pytest.importorskip("Bio")  # structure features parse via biopython
    cfg = _fast_config("fusion")
    df = _toy_frame(40)

    model = train_model(cfg, df)
    assert model.kind == "fusion"

    proba = model.predict_proba(df)
    assert proba.shape == (len(df),)
    assert np.all(np.isfinite(proba))
    assert proba.min() >= 0.0 and proba.max() <= 1.0

    out = model.save(tmp_path / "fusion")
    meta_text = (out / "meta.json").read_text()
    assert '"kind": "fusion"' in meta_text
    assert (out / "weights.pt").exists()

    reloaded = load_model(out)
    assert reloaded.kind == "fusion"
    proba2 = reloaded.predict_proba(df)
    np.testing.assert_allclose(proba, proba2, rtol=1e-4, atol=1e-5)


def test_predict_scores_wrapper() -> None:
    """predict_scores returns a float array aligned to the input rows."""
    cfg = _fast_config("cliff_aware")
    df = _toy_frame(40)
    model = train_model(cfg, df)
    scores = predict_scores(model, df)
    assert scores.dtype == np.float64 or scores.dtype == np.float32
    assert scores.shape == (len(df),)
