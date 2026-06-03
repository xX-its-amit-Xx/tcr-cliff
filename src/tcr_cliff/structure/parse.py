# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Parsing of AlphaFold/ESMFold confidence outputs.

Two ingestion paths are supported:

* :func:`parse_af_json` reads a JSON confidence summary (the schema used by the
  bundled toy examples, with tolerant handling of AlphaFold-3 key variants) into a
  structured :class:`ConfidenceRecord`.
* :func:`parse_pdb_plddt` recovers per-residue pLDDT from the B-factor column of a
  PDB model using Biopython, which is handy when only a folded structure is on disk.

Neither function performs any folding; they only read precomputed artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tcr_cliff._logging import get_logger

_log = get_logger("structure.parse")

# AlphaFold-3 and ESMFold emit the same quantities under slightly different keys.
# These alias lists are searched in order; the first present key wins.
_PTM_KEYS = ("ptm", "pTM", "predicted_tm")
_IPTM_KEYS = ("iptm", "ipTM", "interface_ptm", "predicted_interface_tm")
_PLDDT_KEYS = ("plddt", "pLDDT", "atom_plddts", "plddts")
_PAE_KEYS = ("pae", "paes", "predicted_aligned_error", "pae_matrix")
_TOKEN_CHAIN_KEYS = ("token_chain_ids", "chain_ids", "token_chains")


@dataclass
class ConfidenceRecord:
    """Structured AlphaFold/ESMFold confidence for one TCR-pMHC complex.

    Parameters
    ----------
    ptm:
        Predicted TM-score for the whole complex (global fold confidence).
    iptm:
        Predicted interface TM-score (cross-chain docking confidence). This is the
        single most informative scalar for whether a TCR engages a pMHC.
    plddt:
        Per-token (per-residue) pLDDT confidence, shape ``[L]``.
    pae:
        Predicted aligned error matrix in Angstrom, shape ``[L, L]``.
    token_chain_ids:
        Chain label for every token/residue, length ``L``.
    tcr_chain:
        Chain id that holds the TCR (default ``"A"``).
    pmhc_chains:
        Chain ids that hold the peptide-MHC side (default ``("B",)``).
    """

    ptm: float
    iptm: float
    plddt: np.ndarray
    pae: np.ndarray
    token_chain_ids: list[str]
    tcr_chain: str = "A"
    pmhc_chains: tuple[str, ...] = ("B",)
    extra: dict = field(default_factory=dict)

    @property
    def n_tokens(self) -> int:
        """Number of tokens/residues represented in the record."""
        return len(self.token_chain_ids)

    def interface_mask(self) -> np.ndarray:
        """Boolean ``[L, L]`` mask marking token pairs across the TCR-pMHC interface.

        An entry ``(i, j)`` is ``True`` when one token belongs to the TCR chain and
        the other to a pMHC chain (in either order). The diagonal and intra-chain
        blocks are ``False``.
        """
        chains = np.asarray(self.token_chain_ids, dtype=object)
        is_tcr = chains == self.tcr_chain
        is_pmhc = np.isin(chains, np.asarray(self.pmhc_chains, dtype=object))
        cross = (is_tcr[:, None] & is_pmhc[None, :]) | (is_pmhc[:, None] & is_tcr[None, :])
        return cross


def _first_present(d: dict, keys: tuple[str, ...]) -> object | None:
    """Return ``d[k]`` for the first ``k`` in ``keys`` that exists, else ``None``."""
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return None


def _as_pae_matrix(raw: object, n_tokens: int) -> np.ndarray:
    """Coerce a PAE payload into a square ``[L, L]`` float array.

    AlphaFold-3 occasionally nests the matrix under ``{"predicted_aligned_error": ...}``
    or wraps it in a single-element list; both are handled here.
    """
    if raw is None:
        return np.zeros((n_tokens, n_tokens), dtype=np.float64)
    if isinstance(raw, dict):
        raw = _first_present(raw, _PAE_KEYS)
    arr = np.asarray(raw, dtype=np.float64)
    # Some writers wrap the matrix as ``[[ [..], [..] ]]`` (extra leading axis).
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1 and n_tokens > 0 and arr.size == n_tokens * n_tokens:
        arr = arr.reshape(n_tokens, n_tokens)
    return arr


def parse_af_json(path: str | Path) -> ConfidenceRecord:
    """Parse an AlphaFold/ESMFold confidence JSON into a :class:`ConfidenceRecord`.

    Parameters
    ----------
    path:
        Path to a JSON file with keys ``ptm, iptm, plddt, pae, token_chain_ids`` and
        optionally ``tcr_chain``/``pmhc_chains`` (as in the bundled toy examples).
        AlphaFold-3 key variants (e.g. ``predicted_aligned_error``, ``paes``,
        ``pTM``/``ipTM``) are tolerated.

    Returns
    -------
    ConfidenceRecord
        The parsed confidence record.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    KeyError
        If neither ``plddt`` nor ``token_chain_ids`` can be located.
    """
    path = Path(path)
    payload = json.loads(path.read_text())

    plddt_raw = _first_present(payload, _PLDDT_KEYS)
    chains_raw = _first_present(payload, _TOKEN_CHAIN_KEYS)
    if plddt_raw is None and chains_raw is None:
        raise KeyError(
            f"{path} has neither a pLDDT array nor token_chain_ids; "
            "is this an AlphaFold/ESMFold confidence file?"
        )

    token_chain_ids = [str(c) for c in (chains_raw or [])]
    plddt = np.asarray(plddt_raw if plddt_raw is not None else [], dtype=np.float64)
    n_tokens = len(token_chain_ids) if token_chain_ids else int(plddt.size)
    if not token_chain_ids:
        token_chain_ids = ["A"] * n_tokens

    pae = _as_pae_matrix(_first_present(payload, _PAE_KEYS), n_tokens)

    ptm = float(_first_present(payload, _PTM_KEYS) or 0.0)
    iptm = float(_first_present(payload, _IPTM_KEYS) or 0.0)

    tcr_chain = str(payload.get("tcr_chain", "A"))
    pmhc_raw = payload.get("pmhc_chains", ("B",))
    pmhc_chains = tuple(str(c) for c in pmhc_raw) if pmhc_raw else ("B",)

    _log.debug(
        "parsed %s: L=%d ptm=%.3f iptm=%.3f tcr=%s pmhc=%s",
        path.name,
        n_tokens,
        ptm,
        iptm,
        tcr_chain,
        pmhc_chains,
    )
    return ConfidenceRecord(
        ptm=ptm,
        iptm=iptm,
        plddt=plddt,
        pae=pae,
        token_chain_ids=token_chain_ids,
        tcr_chain=tcr_chain,
        pmhc_chains=pmhc_chains,
    )


def parse_pdb_plddt(path: str | Path) -> dict[str, object]:
    """Read per-residue pLDDT (CA B-factors) and chain ids from a PDB via Biopython.

    AlphaFold/ESMFold store the per-residue pLDDT in the B-factor column of the
    PDB. Reading the CA atom of each residue therefore recovers a per-residue
    confidence trace plus the chain assignment for each residue.

    Parameters
    ----------
    path:
        Path to a PDB file (only the first model is read).

    Returns
    -------
    dict
        ``{"plddt": np.ndarray[L], "chain_ids": list[str]}`` in residue order.

    Raises
    ------
    ImportError
        If Biopython is not installed.
    FileNotFoundError
        If ``path`` does not exist.
    """
    try:
        from Bio.PDB import PDBParser  # lazy: biopython is an optional structure dep
    except ImportError as exc:  # pragma: no cover - exercised only when dep missing
        raise ImportError(
            "parse_pdb_plddt requires Biopython. Install it with "
            "`pip install 'tcr-cliff[structure]'` or `pip install biopython`."
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", str(path))
    model = next(iter(structure))  # first model only

    plddt: list[float] = []
    chain_ids: list[str] = []
    for chain in model:
        for residue in chain:
            if "CA" not in residue:
                continue
            plddt.append(float(residue["CA"].get_bfactor()))
            chain_ids.append(str(chain.id))

    _log.debug("parsed %s: %d CA residues", path.name, len(plddt))
    return {"plddt": np.asarray(plddt, dtype=np.float64), "chain_ids": chain_ids}
