<!--
SPDX-License-Identifier: GPL-3.0-or-later
Copyright (C) 2026 tcr-cliff contributors
-->

# Datasets

`tcr-cliff` ships with a small **synthetic** toy dataset for offline tests and the
cookbook, and provides documented downloaders for four real public TCR–pMHC
resources. This file is the single source of truth for **where the real data comes
from, under what licence, and how to cite it**. Every loader in
`src/tcr_cliff/data/download.py` points failures back here.

> **No data is downloaded or fabricated by the package at import or build time.**
> The loaders fetch on first call and cache to disk; if a fetch or parse fails they
> raise `RuntimeError` referring you to this file so you can download the file
> manually and drop it into the cache directory.

## Cache directory

By default downloads are cached to:

- Linux/macOS: `~/.cache/tcr_cliff/`
- Windows: `C:\tcrtmp\`

Override per call with `load_vdjdb(cache_dir=...)` (and the other loaders), or via
`DataConfig.path`. A cached file is reused on subsequent calls (no re-fetch).

## Real sources

| Dataset | Loader | Source URL (module constant) | Licence | Citation |
| --- | --- | --- | --- | --- |
| **VDJdb** | `load_vdjdb` | `VDJDB_URL` → <https://github.com/antigenomics/vdjdb-db> (`latest-version.txt`) | CC-BY 4.0 | Shugay M. *et al.* "VDJdb: a curated database of T-cell receptor sequences with known antigen specificity." *Nucleic Acids Research* 46(D1):D419–D427, 2018. <https://doi.org/10.1093/nar/gkx760> |
| **McPAS-TCR** | `load_mcpas` | `MCPAS_URL` → <https://friedmanlab.weizmann.ac.il/McPAS-TCR/McPAS-TCR.csv> | Free for academic use (see site terms) | Tickotsky N., Sagiv T., Prilusky J., Shifrut E., Friedman N. "McPAS-TCR: a manually curated catalogue of pathology-associated T cell receptor sequences." *Bioinformatics* 33(18):2924–2929, 2017. <https://doi.org/10.1093/bioinformatics/btx286> |
| **IEDB** (TCR receptor export) | `load_iedb` | `IEDB_URL` → <https://www.iedb.org/database_export_v3.php> | Free, CC0-style; attribution requested (see IEDB terms of use) | Vita R. *et al.* "The Immune Epitope Database (IEDB): 2018 update." *Nucleic Acids Research* 47(D1):D339–D343, 2019. <https://doi.org/10.1093/nar/gky1006> |
| **NetTCR-2.0 / ImmRep** | `load_nettcr` | `NETTCR_URL` → <https://github.com/mnielLab/NetTCR-2.0> (`data/`) | Open (academic; see repository) | Montemurro A. *et al.* "NetTCR-2.0 enables accurate prediction of TCR-peptide binding by using paired TCRα and β sequence data." *Communications Biology* 4:1060, 2021. <https://doi.org/10.1038/s42003-021-02610-3> |

### Notes per source

- **VDJdb** lists *positive* TCR–epitope specificities, so `load_vdjdb` assigns
  `binder = 1`. It filters to TRB (β-chain) records when a gene column is present.
  Generating negatives (e.g. by mispairing) is a downstream modelling choice and is
  intentionally **not** done by the loader. `latest-version.txt` lists the release
  ZIP URLs (newest first); the loader follows the first and reads `vdjdb.slim.txt`.
  *Validated live 2026-06-03: 95,707 β-chain records.*
- **McPAS-TCR** is an association catalogue and is likewise loaded as `binder = 1`.
  The site migrated to an R/Shiny app, so the old `/Download/McPAS-TCR.csv` path 404s,
  but the CSV is still served statically at the app root (the URL above); `load_mcpas`
  downloads it directly (handling the UTF-8 BOM). If the app URL moves again the loader
  detects the HTML response and points you at the manual-download fallback.
  *Validated live 2026-06-03: 15,013 records (356 peptides).*
- **IEDB** is served as a ZIP containing a CSV with a two-row `(group, field)`
  header; the loader flattens it to `"Group - Field"` (so the alpha `Chain 1` and
  beta `Chain 2` CDR3 columns do not collide) and maps `Chain 2 - CDR3 Curated`,
  `Epitope - Name`, and `Assay - MHC Allele Names`. The exact export filename on the
  IEDB portal changes over time — if the constant URL 404s, export the "Receptor
  (TCR/BCR)" table manually from <https://www.iedb.org/database_export_v3.php> and
  place the ZIP at `<cache_dir>/iedb_receptor_full_v3.zip`.
  *Validated live 2026-06-03: 192,167 records.*
- **NetTCR-2.0** ships explicit positive **and** negative pairs, so `load_nettcr`
  preserves the real `0/1` labels. This is the dataset whose
  `CDR3b, peptide, binder` layout `to_nettcr_format` / `write_nettcr_csv` target.
  *Validated live 2026-06-03: 53,952 pairs (8,992 binders).*

If column names in any upstream release drift, the loader raises a `RuntimeError`
listing the columns it actually saw; update the mapping in `download.py` or fetch
the file manually as above.

## Interop with NetTCR-2.0 and ANARCI

- `to_nettcr_format(df)` → a DataFrame with NetTCR-2.0 input columns
  (`CDR3b, peptide, binder`, plus `MHC` when present).
- `write_nettcr_csv(df, path)` → writes that frame to CSV.
- `cdr3_from_anarci(sequence, scheme="imgt")` → extracts the CDR3 loop (IMGT
  positions 105–117) from a full TCR variable-domain sequence. This **lazily
  imports `anarci`**; install it via Bioconda (`conda install -c bioconda anarci`)
  or `pip install ANARCI`. A clear `ImportError` is raised if it is missing.

## Bundled toy set is SYNTHETIC

The dataset under `src/tcr_cliff/data/toy/` (`toy_pairs.csv` and the
`structure/` JSON/PDB examples) is **synthetic**. It is produced by
[`src/tcr_cliff/data/toy/_generate.py`](src/tcr_cliff/data/toy/_generate.py),
which embeds a simplified, learnable binding rule with deliberately constructed
activity cliffs (single-residue anchor substitutions that flip the label) and
matched smooth controls, plus structure-confidence values correlated with binding.

> **The toy set is for offline tests, demos, and the cookbook only. It contains no
> real experimental measurements and must NOT be used for any benchmark claim,
> publication figure, or model comparison reported as scientific evidence.**

For real benchmarking, use the documented downloaders above with proper
peptide/CDR3-grouped splits (see `split_pairs`) to avoid leakage.
