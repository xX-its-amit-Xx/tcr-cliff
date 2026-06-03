# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Generate the bundled **synthetic** toy TCR-pMHC dataset.

IMPORTANT — provenance and honesty
----------------------------------
This script produces a *synthetic*, clearly-labelled toy dataset for offline tests
and the cookbook. It is NOT real experimental data and must never be used for
benchmark claims. Real data is obtained through the documented downloaders in
:mod:`tcr_cliff.data.download` (VDJdb, McPAS-TCR, IEDB, NetTCR/ImmRep); see
``DATASETS.md``.

Design
------
We embed a *learnable* binding rule with explicit activity cliffs so that the
cliff-aware evaluation, the cliff-aware model, the structure-fusion lift, and the
anchor-position interpretation all have genuine signal to recover:

* Binding depends on classic HLA-A*02:01-style peptide anchors (P2 and the
  C-terminus PΩ) plus a CDR3-beta hotspot motif. This is biologically motivated
  but deliberately simplified.
* **Cliffs** are created by single-residue substitutions at an anchor position
  that flip the binding label (edit distance 1, opposite outcome) — the altered
  peptide ligand phenomenon the package targets.
* **Smooth neighbours** are single-residue substitutions at non-anchor positions
  that preserve the label (controls, so cliffs are not trivially "any mutation").
* Structural confidence (ipTM, interface PAE, pLDDT) is correlated with true
  binding, so the fusion model can show a measurable lift.

Run ``python -m tcr_cliff.data.toy._generate`` to regenerate the CSV/JSON files.
The output is deterministic (fixed seed).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
AA = "ACDEFGHIKLMNPQRSTVWY"

# Real, public HLA class-I allele *names* (not fabricated measurements) used as labels.
MHC_ALLELES = ["HLA-A*02:01", "HLA-A*01:01", "HLA-B*07:02", "HLA-A*24:02"]
# A short, illustrative MHC pseudo-sequence per allele (NetMHCpan-style 34-mer slots,
# truncated here for the toy set; real pseudo-sequences come from the loaders).
MHC_PSEUDO = {
    "HLA-A*02:01": "YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY",
    "HLA-A*01:01": "YFAMYQENMAHTDANTLYIIYRDYTWVARVYRGY",
    "HLA-B*07:02": "YYATYRNIFTNTYESNLYLRYDSYTWAEWAYTWY",
    "HLA-A*24:02": "YSAMYEEKVAHTDENIAYLMYHDYTWAVWAYTWY",
}

# Anchor preferences (HLA-A*02:01-like): hydrophobic P2 and PΩ favour binding.
GOOD_P2 = set("LMIV")
GOOD_POMEGA = set("VLIA")
# CDR3-beta hotspot: a positively contributing central motif residue set.
CDR3_HOTSPOT = set("RKWY")


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _rand_peptide(rng: np.random.Generator, length: int = 9) -> str:
    return "".join(rng.choice(list(AA), size=length))


def _rand_cdr3(rng: np.random.Generator) -> str:
    length = int(rng.integers(12, 17))
    core = "".join(rng.choice(list(AA), size=length - 2))
    return "C" + core + "F"  # canonical CDR3 flanks


def _binding_score(cdr3b: str, peptide: str) -> float:
    """Latent binding score; > 0 means binder. Drives label, affinity, ipTM."""
    s = -1.0
    if peptide[1] in GOOD_P2:
        s += 1.4
    if peptide[-1] in GOOD_POMEGA:
        s += 1.3
    # central CDR3 hotspot contribution
    mid = cdr3b[len(cdr3b) // 2 - 1 : len(cdr3b) // 2 + 1]
    s += 0.9 * sum(c in CDR3_HOTSPOT for c in mid)
    # mild complementarity term: a charged peptide P5 likes an aromatic CDR3 hotspot
    if peptide[4] in "DE" and any(c in "WYF" for c in cdr3b):
        s += 0.5
    return s


def _is_binder(cdr3b: str, peptide: str) -> int:
    return int(_binding_score(cdr3b, peptide) > 0.0)


def _flip_anchor(rng: np.random.Generator, peptide: str, want_binder: int) -> str | None:
    """Mutate an anchor (P2 or PΩ) to flip the label; return None if it cannot."""
    positions = [1, len(peptide) - 1]
    rng.shuffle(positions)
    for pos in positions:
        good = GOOD_P2 if pos == 1 else GOOD_POMEGA
        choices = list(good) if want_binder else [a for a in AA if a not in good]
        rng.shuffle(choices)
        for aa in choices:
            if aa == peptide[pos]:
                continue
            mut = peptide[:pos] + aa + peptide[pos + 1 :]
            return mut
    return None


def _mutate_nonanchor(rng: np.random.Generator, peptide: str) -> str | None:
    """Mutate a non-anchor position (a smooth neighbour candidate)."""
    interior = [i for i in range(len(peptide)) if i not in (1, len(peptide) - 1)]
    rng.shuffle(interior)
    for pos in interior:
        choices = [a for a in AA if a != peptide[pos]]
        aa = rng.choice(choices)
        return peptide[:pos] + aa + peptide[pos + 1 :]
    return None


def _affinity(rng: np.random.Generator, score: float, binder: int) -> float:
    """-log10(Kd)-style value: higher = stronger. Correlated with score + noise."""
    base = 5.0 + 1.1 * score + rng.normal(0, 0.4)
    if not binder:
        base -= 1.5
    return float(np.clip(base, 3.0, 11.0))


def generate(seed: int = 7, n_anchor_families: int = 90, n_random: int = 160) -> pd.DataFrame:
    """Build the toy pairs table with cliff and smooth-neighbour structure."""
    rng = _rng(seed)
    rows: list[dict] = []
    pid = 0

    def add(cdr3b, peptide, mhc, binder, score, group):
        nonlocal pid
        rows.append(
            {
                "pair_id": f"toy{pid:04d}",
                "cdr3b": cdr3b,
                "peptide": peptide,
                "mhc": mhc,
                "mhc_pseudo": MHC_PSEUDO[mhc],
                "v_gene": "TRBV" + str(int(rng.integers(2, 29))),
                "j_gene": "TRBJ"
                + str(int(rng.integers(1, 3)))
                + "-"
                + str(int(rng.integers(1, 8))),
                "binder": binder,
                "affinity": _affinity(rng, score, binder),
                "source": "synthetic_toy",
                "_group": group,
            }
        )
        pid += 1

    # 1) Anchor families: a seed peptide + a single-residue cliff partner (label flip)
    #    + a smooth single-residue neighbour (label preserved).
    for fam in range(n_anchor_families):
        mhc = MHC_ALLELES[fam % len(MHC_ALLELES)]
        cdr3b = _rand_cdr3(rng)
        # Seed an unambiguous binder or non-binder.
        for _ in range(50):
            pep = _rand_peptide(rng)
            seed_binder = _is_binder(cdr3b, pep)
            if abs(_binding_score(cdr3b, pep)) > 0.6:  # avoid borderline seeds
                break
        add(cdr3b, pep, mhc, seed_binder, _binding_score(cdr3b, pep), f"fam{fam}")

        # Cliff partner: flip the label via an anchor substitution.
        cliff = _flip_anchor(rng, pep, want_binder=1 - seed_binder)
        if cliff is not None and _is_binder(cdr3b, cliff) == (1 - seed_binder):
            add(cdr3b, cliff, mhc, 1 - seed_binder, _binding_score(cdr3b, cliff), f"fam{fam}")

        # Smooth neighbour: non-anchor substitution preserving the label.
        smooth = _mutate_nonanchor(rng, pep)
        if smooth is not None and _is_binder(cdr3b, smooth) == seed_binder:
            add(cdr3b, smooth, mhc, seed_binder, _binding_score(cdr3b, smooth), f"fam{fam}")

    # 2) Random background pairs for breadth and class balance.
    for _ in range(n_random):
        mhc = MHC_ALLELES[int(rng.integers(0, len(MHC_ALLELES)))]
        cdr3b = _rand_cdr3(rng)
        pep = _rand_peptide(rng)
        b = _is_binder(cdr3b, pep)
        add(cdr3b, pep, mhc, b, _binding_score(cdr3b, pep), "background")

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["cdr3b", "peptide", "mhc"]).reset_index(drop=True)

    # Grouped train/val/test split on the peptide family so cliff partners stay together
    # and no peptide leaks across splits.
    groups = df["_group"].to_numpy()
    uniq = sorted(set(groups))
    rng.shuffle(uniq)
    n = len(uniq)
    test_g = set(uniq[: int(0.2 * n)])
    val_g = set(uniq[int(0.2 * n) : int(0.3 * n)])
    split = np.where(
        np.isin(groups, list(test_g)),
        "test",
        np.where(np.isin(groups, list(val_g)), "val", "train"),
    )
    df["split"] = split
    return df.drop(columns=["_group"])


def _confidence_for(rng: np.random.Generator, cdr3b: str, peptide: str, binder: int) -> dict:
    """Synthesise AlphaFold/ESMFold-style confidence correlated with binding."""
    n_tcr, n_pep = len(cdr3b), len(peptide)
    n = n_tcr + n_pep
    # ipTM/pTM: binders fold to a more confident interface.
    iptm = float(np.clip(rng.normal(0.72 if binder else 0.42, 0.08), 0.05, 0.95))
    ptm = float(np.clip(rng.normal(0.78 if binder else 0.6, 0.06), 0.1, 0.95))
    # pLDDT per residue: high within chains; binders slightly higher at interface.
    plddt = np.clip(rng.normal(82 if binder else 74, 6, size=n), 30, 99)
    # PAE matrix: confident intra-chain blocks; inter-chain block lower for non-binders.
    pae = rng.uniform(2, 6, size=(n, n)).astype(float)
    inter = float(np.clip(rng.normal(6 if binder else 13, 1.5), 1.0, 30.0))
    pae[:n_tcr, n_tcr:] = rng.normal(inter, 1.0, size=(n_tcr, n_pep)).clip(0.5, 31.0)
    pae[n_tcr:, :n_tcr] = pae[:n_tcr, n_tcr:].T
    np.fill_diagonal(pae, 0.4)
    chain_ids = ["A"] * n_tcr + ["B"] * n_pep
    return {
        "ptm": round(ptm, 4),
        "iptm": round(iptm, 4),
        "plddt": [round(float(x), 2) for x in plddt],
        "pae": [[round(float(v), 2) for v in row] for row in pae],
        "token_chain_ids": chain_ids,
        "tcr_chain": "A",
        "pmhc_chains": ["B"],
    }


def _summary_row(rec: dict, cutoff: float = 8.0) -> dict:
    """Derive compact interface features from a full confidence record."""
    chain = np.array(rec["token_chain_ids"])
    pae = np.array(rec["pae"])
    plddt = np.array(rec["plddt"])
    tcr = chain == rec["tcr_chain"]
    pmhc = np.isin(chain, rec["pmhc_chains"])
    inter_pae = pae[np.ix_(tcr, pmhc)]
    contacts = inter_pae < cutoff
    return {
        "ptm": rec["ptm"],
        "iptm": rec["iptm"],
        "plddt_mean": round(float(plddt.mean()), 3),
        "plddt_min": round(float(plddt.min()), 3),
        "pae_interface_mean": round(float(inter_pae.mean()), 3),
        "pae_interface_min": round(float(inter_pae.min()), 3),
        "n_interface_contacts": int(contacts.sum()),
        "plddt_interface_mean": round(float(plddt[pmhc].mean()), 3),
    }


def write_all(seed: int = 7) -> None:
    """Generate and write the toy CSV plus structure-confidence files."""
    df = generate(seed=seed)
    df.to_csv(HERE / "toy_pairs.csv", index=False)

    struct_dir = HERE / "structure"
    examples_dir = struct_dir / "examples"
    struct_dir.mkdir(exist_ok=True)
    examples_dir.mkdir(exist_ok=True)

    rng = _rng(seed + 1)
    summary: dict[str, dict] = {}
    # A few full example JSONs (binder + its cliff partner) to exercise the parser.
    example_ids = list(df["pair_id"].iloc[:4]) + list(df["pair_id"].iloc[10:12])
    for _, row in df.iterrows():
        rec = _confidence_for(rng, row["cdr3b"], row["peptide"], int(row["binder"]))
        summary[row["pair_id"]] = _summary_row(rec)
        if row["pair_id"] in example_ids:
            (examples_dir / f"{row['pair_id']}.json").write_text(json.dumps(rec, indent=2))

    (struct_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_example_pdb(examples_dir / "example.pdb", df.iloc[0])
    n_pos = int(df["binder"].sum())
    print(
        f"Wrote {len(df)} pairs ({n_pos} binders / {len(df) - n_pos} non-binders), "
        f"{len(summary)} confidence summaries, {len(example_ids)} full examples."
    )


def _write_example_pdb(path: Path, row: pd.Series) -> None:
    """Write a minimal CA-only PDB with pLDDT encoded in the B-factor column."""
    rng = _rng(123)
    lines = ["REMARK  Synthetic toy structure; B-factor column holds pLDDT.", "MODEL        1"]
    serial = 1
    seqs = [("A", row["cdr3b"]), ("B", row["peptide"])]
    aa3 = {
        "A": "ALA",
        "C": "CYS",
        "D": "ASP",
        "E": "GLU",
        "F": "PHE",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "K": "LYS",
        "L": "LEU",
        "M": "MET",
        "N": "ASN",
        "P": "PRO",
        "Q": "GLN",
        "R": "ARG",
        "S": "SER",
        "T": "THR",
        "V": "VAL",
        "W": "TRP",
        "Y": "TYR",
    }
    for chain, seq in seqs:
        for i, aa in enumerate(seq, start=1):
            x, y, z = (rng.uniform(-20, 20) for _ in range(3))
            plddt = float(np.clip(rng.normal(80, 8), 30, 99))
            lines.append(
                f"ATOM  {serial:5d}  CA  {aa3.get(aa, 'GLY')} {chain}{i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{plddt:6.2f}           C"
            )
            serial += 1
    lines += ["ENDMDL", "END"]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":  # pragma: no cover
    write_all()
