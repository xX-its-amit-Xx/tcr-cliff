# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Fixed-length structural-confidence feature vectors for the fusion model.

A folded TCR-pMHC complex is reduced to a 12-dimensional confidence fingerprint so
the fusion head can concatenate it onto sequence embeddings. The vector layout is
**fixed** (see :data:`STRUCTURE_FEATURE_NAMES`): the first eight entries mirror the
precomputed summary schema, the last four are cheap derived signals, and the final
``available_flag`` lets a model learn to down-weight rows with no structure.
"""

from __future__ import annotations

import numpy as np

from tcr_cliff._logging import get_logger
from tcr_cliff.structure.parse import ConfidenceRecord

_log = get_logger("structure.features")

# The 12 feature names, in their fixed column order. The first eight match the
# bundled summary.json keys exactly; the last four are derived. Length MUST be 12.
STRUCTURE_FEATURE_NAMES: list[str] = [
    "ptm",
    "iptm",
    "plddt_mean",
    "plddt_min",
    "pae_interface_mean",
    "pae_interface_min",
    "n_interface_contacts",
    "plddt_interface_mean",
    "iptm_x_ptm",
    "contact_density",
    "pae_interface_range",
    "available_flag",
]

# Summary keys consumed by :func:`features_from_summary`, in their canonical order.
_SUMMARY_KEYS: tuple[str, ...] = (
    "ptm",
    "iptm",
    "plddt_mean",
    "plddt_min",
    "pae_interface_mean",
    "pae_interface_min",
    "n_interface_contacts",
    "plddt_interface_mean",
)

# Number of interface contacts used to normalise ``contact_density`` to ~[0, 1].
_CONTACT_NORM: float = 100.0

assert len(STRUCTURE_FEATURE_NAMES) == 12, "STRUCTURE_FEATURE_NAMES must have length 12"


def _assemble(
    ptm: float,
    iptm: float,
    plddt_mean: float,
    plddt_min: float,
    pae_interface_mean: float,
    pae_interface_min: float,
    n_interface_contacts: float,
    plddt_interface_mean: float,
    available: bool,
) -> np.ndarray:
    """Build the 12-vector from the eight base scalars plus an availability flag."""
    iptm_x_ptm = iptm * ptm
    contact_density = n_interface_contacts / _CONTACT_NORM
    pae_interface_range = pae_interface_mean - pae_interface_min
    vec = np.array(
        [
            ptm,
            iptm,
            plddt_mean,
            plddt_min,
            pae_interface_mean,
            pae_interface_min,
            n_interface_contacts,
            plddt_interface_mean,
            iptm_x_ptm,
            contact_density,
            pae_interface_range,
            1.0 if available else 0.0,
        ],
        dtype=np.float64,
    )
    return vec


def missing_vector() -> np.ndarray:
    """Return the all-zero feature vector for records with no structure.

    The 12 values are zero except ``available_flag`` which is ``0.0`` too, so the
    whole vector is zeros. Models can key off ``available_flag`` to ignore the block.

    Returns
    -------
    np.ndarray
        Zeros of shape ``[12]``.
    """
    return np.zeros(len(STRUCTURE_FEATURE_NAMES), dtype=np.float64)


def features_from_summary(d: dict | None) -> np.ndarray:
    """Build the 12-feature vector from a precomputed summary dict.

    Parameters
    ----------
    d:
        Mapping with keys ``ptm, iptm, plddt_mean, plddt_min, pae_interface_mean,
        pae_interface_min, n_interface_contacts, plddt_interface_mean``. Missing
        keys default to ``0.0``. ``None`` yields the missing vector.

    Returns
    -------
    np.ndarray
        Shape ``[12]`` following :data:`STRUCTURE_FEATURE_NAMES`. ``available_flag``
        is ``1.0`` for a present summary, ``0.0`` for ``None``.
    """
    if d is None:
        return missing_vector()
    vals = {k: float(d.get(k, 0.0)) for k in _SUMMARY_KEYS}
    return _assemble(**vals, available=True)


def features_from_record(rec: ConfidenceRecord | None, cutoff: float = 8.0) -> np.ndarray:
    """Compute the 12-feature vector from a parsed :class:`ConfidenceRecord`.

    Interface quantities are derived from the cross-chain PAE block: a token pair is
    an *interface contact* when its predicted aligned error is below ``cutoff`` (in
    Angstrom). pLDDT statistics are taken over all tokens, and ``plddt_interface_mean``
    over the tokens that participate in at least one interface contact.

    Parameters
    ----------
    rec:
        Parsed confidence record, or ``None`` for a missing structure.
    cutoff:
        PAE threshold (Angstrom) below which a cross-chain token pair counts as an
        interface contact. Defaults to ``8.0``.

    Returns
    -------
    np.ndarray
        Shape ``[12]`` following :data:`STRUCTURE_FEATURE_NAMES`.
    """
    if rec is None:
        return missing_vector()

    plddt = np.asarray(rec.plddt, dtype=np.float64)
    plddt_mean = float(plddt.mean()) if plddt.size else 0.0
    plddt_min = float(plddt.min()) if plddt.size else 0.0

    pae = np.asarray(rec.pae, dtype=np.float64)
    iface = rec.interface_mask()

    if pae.shape == iface.shape and iface.any():
        cross_pae = pae[iface]
        contact = iface & (pae < cutoff)
        n_contacts = int(contact.sum())
        if n_contacts > 0:
            contact_pae = pae[contact]
            pae_interface_mean = float(contact_pae.mean())
            pae_interface_min = float(contact_pae.min())
            # Tokens touching at least one interface contact.
            contact_tokens = contact.any(axis=1) | contact.any(axis=0)
            plddt_interface_mean = (
                float(plddt[contact_tokens].mean())
                if plddt.size and contact_tokens.any()
                else plddt_mean
            )
        else:
            # No sub-cutoff contacts: summarise the whole cross-chain block instead.
            pae_interface_mean = float(cross_pae.mean())
            pae_interface_min = float(cross_pae.min())
            plddt_interface_mean = plddt_mean
    else:
        n_contacts = 0
        pae_interface_mean = float(pae[pae > 0].mean()) if (pae > 0).any() else 0.0
        pae_interface_min = float(pae[pae > 0].min()) if (pae > 0).any() else 0.0
        plddt_interface_mean = plddt_mean

    return _assemble(
        ptm=float(rec.ptm),
        iptm=float(rec.iptm),
        plddt_mean=plddt_mean,
        plddt_min=plddt_min,
        pae_interface_mean=pae_interface_mean,
        pae_interface_min=pae_interface_min,
        n_interface_contacts=float(n_contacts),
        plddt_interface_mean=plddt_interface_mean,
        available=True,
    )
