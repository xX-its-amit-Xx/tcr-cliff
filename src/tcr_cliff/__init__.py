# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""tcr-cliff: activity-cliff-aware TCR-pMHC binding prediction.

The package is organised into loosely coupled submodules:

* :mod:`tcr_cliff.data`        - dataset loaders, downloaders, and the bundled toy set.
* :mod:`tcr_cliff.embeddings`  - ESM-2 / fallback sequence embeddings with disk caching.
* :mod:`tcr_cliff.cliffs`      - binding-cliff definition, detection, and cliff graphs.
* :mod:`tcr_cliff.structure`   - AlphaFold/ESMFold confidence ingestion (pTM, ipTM, PAE).
* :mod:`tcr_cliff.models`      - LightGBM baseline + cliff-aware contrastive PyTorch model.
* :mod:`tcr_cliff.interpret`   - SHAP and per-residue attribution onto CDR3/peptide anchors.
* :mod:`tcr_cliff.eval`        - standard metrics plus the headline cliff-aware evaluation.

Heavy optional dependencies (torch, fair-esm, lightgbm) are imported lazily so that
``import tcr_cliff`` and the offline toy tests run on a minimal install.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Lightweight, dependency-free symbols are re-exported eagerly. Anything that may
# pull in torch / fair-esm / lightgbm is imported lazily inside its own submodule.
from tcr_cliff._logging import get_logger, setup_logging
from tcr_cliff.config import Config, load_config

__all__ = [
    "Config",
    "__version__",
    "get_logger",
    "load_config",
    "setup_logging",
]
