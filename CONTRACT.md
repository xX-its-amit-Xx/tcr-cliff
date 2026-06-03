# tcr-cliff internal build contract

This document is the authoritative interface spec for the package. The **spine**
(data, config, embeddings, cliffs, eval, models baseline+registry) is already
implemented and tested. Build the remaining leaf modules **against these fixed
interfaces** so everything composes.

## Conventions (MANDATORY for every file you create)
- First two lines of every `.py`:
  ```python
  # SPDX-License-Identifier: GPL-3.0-or-later
  # Copyright (C) 2026 tcr-cliff contributors
  ```
- `from __future__ import annotations` at the top. Full type hints + NumPy-style docstrings.
- Use `from tcr_cliff._logging import get_logger`; **never `print`** in library code
  (Typer CLI user output and the manual toy generator are the only exceptions).
- Heavy optional deps (`torch`, `fair-esm`, `transformers`, `lightgbm`, `shap`,
  `anarci`) must be imported **lazily inside functions/methods**, never at module top,
  and must raise a clear, actionable error if missing. GPU is optional: default
  `device="cpu"`, detect CUDA only if available.
- Line length 100. Code must pass `ruff check` and `black`.
- **Disk is constrained** on this machine. Do not write large artifacts. If you must
  cache, use `C:\tcrtmp`. Do NOT run heavy torch training or download models/data
  locally — only light validation (`python -c "import ast"` / `py_compile` / importing
  your module). Heavy integration is run by the orchestrator afterward.
- Tests you write must **skip** (via `pytest.importorskip`) when their heavy dep is
  absent, so the offline suite stays green. fair-esm, lightgbm, and network are absent
  here; torch/transformers/shap ARE installed.

## Package layout
```
src/tcr_cliff/
  __init__.py config.py _logging.py utils/         # DONE
  data/   schema.py loaders.py toy/ __init__.py     # DONE  (download.py TO BUILD)
  embeddings/ base.py fallback.py cache.py __init__ # DONE  (esm.py TO BUILD)
  cliffs/ distance.py detect.py graph.py stats.py   # DONE
  structure/                                        # TO BUILD
  models/ features.py baseline.py registry.py __init__  # DONE (contrastive/dataset/head/fusion TO BUILD)
  interpret/                                         # TO BUILD
  eval/   metrics.py cliff_eval.py report.py __init__   # DONE
  cli/                                               # TO BUILD
```

## Canonical data schema (`tcr_cliff.data.schema`)
A normalised pairs `DataFrame` has columns:
`pair_id, cdr3b, peptide, mhc, mhc_pseudo, v_gene, j_gene, binder(0/1 int),
affinity(float, NaN ok), source, split`.
`AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"`. `validate_pairs(df)` normalises/validates.
Toy data: `from tcr_cliff.data import load_toy, load_toy_structure_summary, split_pairs, toy_path`.
`toy_path("structure/summary.json")`, `toy_path("structure/examples/<id>.json")`,
`toy_path("structure/examples/example.pdb")` exist.

## Config (`tcr_cliff.config`)
`Config` has: `seed, output_dir, data(DataConfig), embedding(EmbeddingConfig),
cliff(CliffConfig), structure(StructureConfig), model(ModelConfig), train(TrainConfig),
eval(EvalConfig)`. `load_config(path=None, **overrides) -> Config`. `cfg.to_yaml(path)`.
Key fields you will read:
- `EmbeddingConfig`: backend('fallback'|'esm-fair'|'esm-hf'), model_name, pooling('mean'|'per_residue'),
  layers(list[int]), include_mhc(bool), cache_dir(Path|None), device, batch_size, max_len,
  fallback_dim, kmer_k; property `.fields -> tuple[str,...]`.
- `CliffConfig`: max_edits(k), distance('levenshtein'|'hamming'), vary('peptide'|'cdr3b'|'both'),
  require_label_flip(bool), affinity_delta(float|None), same_context(bool), min_len.
- `StructureConfig`: enabled(bool), source_dir(Path|None), use_stub(bool), pae_interface_cutoff(float=8.0).
- `ModelConfig`: kind('baseline_lgbm'|'cliff_aware'|'fusion'), baseline(BaselineModelConfig),
  cliff_aware(CliffAwareModelConfig), fusion(FusionModelConfig).
- `CliffAwareModelConfig`: proj_dim, hidden_dim, dropout, contrastive_epochs, contrastive_margin,
  contrastive_temperature, cliff_weight, head_epochs, lr, weight_decay, batch_size,
  freeze_encoder_after_pretrain.
- `FusionModelConfig`: structure_feature_dim(12), hidden_dim, dropout.

## Embeddings (`tcr_cliff.embeddings`)  — CONSUME these
- `get_embedder(cfg.embedding) -> Embedder`
- `embed_pairs(df, cfg.embedding, fields=None, pooling=None) -> dict[str, EmbeddingResult]`
- `build_feature_matrix(df, cfg.embedding, fields=None) -> (np.ndarray[N,D], names:list[str])`
- `Embedder` ABC: `.embed(seqs, pooling='mean'|'per_residue') -> EmbeddingResult`; subclass implements
  `dim` property, `_embed_mean(seqs)->[N,D]`, optional `_embed_per_residue(seqs)->list[[L,d]]`.
- `EmbeddingResult(sequences, backend, dim, vectors=[N,D], per_residue=list)`.

## Cliffs (`tcr_cliff.cliffs`) — CONSUME these
- `find_neighbor_pairs(df, cfg.cliff) -> list[NeighborPair]`; `find_cliffs(df, cfg.cliff)`.
- `NeighborPair` fields: `i, j` (positional row indices into df), `id_i, id_j, vary, seq_i, seq_j,
  distance, label_i, label_j, affinity_i, affinity_j, is_cliff, reason`.
- `build_cliff_graph(pairs, df=None, cliffs_only=False) -> nx.Graph`, `cliff_components`, `cliff_hubs`,
  `export_graph`. `cliff_statistics(pairs, n_records) -> dict`. `mark_cliff_membership(df, pairs)`
  adds `in_cliff`/`in_neighbor` bool columns.

## Eval (`tcr_cliff.eval`) — CONSUME these
- `binary_metrics(y_true, y_score, threshold=0.5) -> dict(auroc,auprc,accuracy,f1,precision,recall,n,n_pos)`.
- `regression_metrics(y_true, y_pred) -> dict(pearson,spearman,rmse,mae,n)`.
- `cliff_aware_report(df_test, y_score, pairs=None, cliff_cfg=None, threshold=0.5)
  -> {overall, cliff_records, non_cliff_records, pair_level, gap}` (y_score aligned to df rows).
- `compare_models({name: report}) -> DataFrame`; `cliff_report_to_frame(report)`; `save_report`.

## Models (`tcr_cliff.models`) — CONSUME registry; BUILD torch models
- Registry: `train_model(cfg, df_train, df_val=None) -> Model`, `load_model(path) -> Model`,
  `predict_scores(model, df) -> np.ndarray`. `Model` protocol = `predict_proba(df)->np.ndarray`,
  `save(dir)->Path`, classmethod `load(dir)`, attribute `kind:str`.
- Feature assembly: `sequence_features(df, cfg) -> (X,names)`,
  `structure_features(df, cfg) -> (X,names)` (lazy-imports structure),
  `assemble_features(df, cfg, with_structure=False) -> (X, names)`.
- Saved models are **directories** containing `meta.json` (`{"kind": ...}`), payload, `config.yaml`.

---

# MODULES TO BUILD (exact required interfaces)

### embeddings/esm.py
`class ESMEmbedder(Embedder)` with
`__init__(self, model_name="esm2_t12_35M_UR50D", backend="esm-hf", device="cpu",
layers=(-1,), batch_size=8, max_len=64)`. Implements `dim`, `_embed_mean`, `_embed_per_residue`.
- `backend="esm-fair"` -> use `import esm` (fair-esm): load via `esm.pretrained.<model_name>()`,
  batch through the alphabet's `BatchConverter`, take representations at `layers`, mean-pool over
  residues (excluding BOS/EOS) for mean; return per-residue token reps otherwise.
- `backend="esm-hf"` -> use `transformers.AutoTokenizer`/`EsmModel`. Map fair-esm style names to
  HF ids (e.g. `esm2_t12_35M_UR50D` -> `facebook/esm2_t12_35M_UR50D`); accept either form.
- Lazy imports; clear ImportError message naming the extra (`pip install 'tcr-cliff[esm]'`).
  `torch.no_grad()`, `.eval()`, move to device if cuda available, batch by `batch_size`.
- Set `self.name = f"esm-{backend}-{model_name}-L{layers}"` for cache keys.
Write `tests/test_embeddings_esm.py` using `pytest.importorskip("esm")` /
`importorskip("transformers")` and mark `@pytest.mark.esm` (these will skip here).

### structure/  (parse.py, features.py, __init__.py)
- `parse.py`: `@dataclass ConfidenceRecord(ptm:float, iptm:float, plddt:np.ndarray,
  pae:np.ndarray, token_chain_ids:list[str], tcr_chain:str="A", pmhc_chains:tuple=("B",))`.
  `parse_af_json(path) -> ConfidenceRecord` (AlphaFold/ESMFold JSON: keys `ptm,iptm,plddt,pae,
  token_chain_ids,tcr_chain,pmhc_chains` as in toy examples; tolerate AF3 key variants like
  `predicted_aligned_error`, `pae`/`paes`). `parse_pdb_plddt(path) -> dict(plddt:np.ndarray,
  chain_ids:list[str])` reading B-factors of CA atoms via biopython.
- `features.py`: module constant `STRUCTURE_FEATURE_NAMES: list[str]` of length **12**.
  `features_from_record(rec, cutoff=8.0) -> np.ndarray[12]` and
  `features_from_summary(d: dict) -> np.ndarray[12]` where summary dict keys are exactly:
  `ptm, iptm, plddt_mean, plddt_min, pae_interface_mean, pae_interface_min,
  n_interface_contacts, plddt_interface_mean`. The 12 features = those 8 (in that order) +
  4 derived: `iptm_x_ptm`, `contact_density (=n_interface_contacts normalised, e.g. /100)`,
  `pae_interface_range (=mean-min)`, `available_flag (1.0)`. Missing record -> zeros with
  `available_flag=0.0`.
- `__init__.py`: `load_structure_features(df, structure_cfg) -> (np.ndarray[N,12], names)`.
  When `structure_cfg.use_stub` (default) load `tcr_cliff.data.load_toy_structure_summary()`
  and look up by `pair_id`; rows without an entry get the missing vector. When `source_dir` is
  set, read `<source_dir>/<pair_id>.json` via `parse_af_json` + `features_from_record`.
  Re-export `parse_af_json, parse_pdb_plddt, ConfidenceRecord, STRUCTURE_FEATURE_NAMES`.
Write `tests/test_structure.py` (no skip needed; biopython is installed): parse the bundled
example JSON + PDB, assert feature vector length 12, assert binder example has higher iptm than a
non-binder example, assert `load_structure_features(toy_df, StructureConfig())` returns `[N,12]`.

### models/ (dataset.py, contrastive.py, head.py, fusion.py)  — torch; lazy
- `dataset.py`: helpers to build per-record feature tensors and contrastive pair batches.
  `build_pair_index(df, cfg) -> (pairs:list[NeighborPair])` using `find_neighbor_pairs`.
  `class ContrastivePairs(torch.utils.data.Dataset)` yielding `(x_i, x_j, is_cliff_float)` where
  x are rows of `sequence_features(df,cfg)`. A `collate`/loader helper. Import torch lazily at use.
- `contrastive.py`: `class CliffAwareEncoder(nn.Module)` MLP `in_dim -> hidden_dim -> proj_dim`
  (ReLU, dropout, L2-normalised output). `contrastive_pretrain(encoder, X, pairs, cfg) -> encoder`
  implementing the cliff-aware objective: **pull smooth (non-cliff) neighbour pairs together, push
  cliff pairs apart** in embedding space — a margin/contrastive loss with `cliff_weight` upweighting
  the push term and `contrastive_margin`/`contrastive_temperature` from cfg. Seed via
  `tcr_cliff.utils.seed.seed_everything(cfg.train.seed)`. Log losses, no prints.
- `head.py`: `class CliffAwareModel` conforming to the Model protocol:
  `__init__(cfg)`, `fit(df_train, df_val=None) -> self` (computes `sequence_features`, builds pairs,
  runs `contrastive_pretrain`, then trains a supervised `BindingHead(nn.Module)` =
  encoder(optionally frozen) + linear classifier with BCE), `predict_proba(df) -> np.ndarray`,
  `save(dir) -> Path` (writes `meta.json{"kind":"cliff_aware"}`, `weights.pt`, `config.yaml`,
  and the input feature dim), classmethod `load(dir)`. `kind="cliff_aware"`.
- `fusion.py`: `class FusionModel` (Model protocol, `kind="fusion"`): like CliffAwareModel but the
  binding head input is `concat(sequence_rep, structure_features(df,cfg))`; reuses the cliff-aware
  encoder for the sequence part. `save/load` mirror cliff_aware with `meta.json{"kind":"fusion"}`.
Write `tests/test_models_torch.py` with `pytest.importorskip("torch")` (torch IS installed, so keep
it FAST: fallback_dim<=64, contrastive_epochs<=2, head_epochs<=3, toy `head(40)` rows) verifying
fit/predict_proba shape in [0,1] and save/load round-trip for both cliff_aware and fusion. Keep
total runtime under ~30s and memory small.

### interpret/ (shap_baseline.py, attribution.py, anchors.py, plots.py, __init__.py)
- `shap_baseline.py`: `explain_baseline(model: BaselineModel, df, max_display=20, background=None)
  -> dict` with SHAP values over the embedding features, aggregated back to per-field blocks
  (`cdr3b`,`peptide`,`mhc_pseudo`) by summing |shap| within each field's columns
  (use `model.feature_names`, names look like `"<field>_<j>"`). `shap.TreeExplainer` for LightGBM;
  for sklearn HistGBM fall back to `shap.Explainer` or permutation importance. Lazy import shap.
- `attribution.py`: **model-agnostic occlusion** so it works without torch:
  `residue_attribution(predict_fn, row: pd.Series, field="peptide", cfg=None) -> np.ndarray[L]`
  where `predict_fn(df)->scores`; for each residue, substitute alanine (or mask) and measure the
  drop in predicted score for that single record. Also `attention_rollout(model, df, ...)` for the
  torch model when available (lazy torch; optional). Return per-residue importance arrays.
- `anchors.py`: `anchor_importance(attr: np.ndarray, peptide: str) -> dict` mapping importance onto
  canonical anchor positions P2 and PΩ (C-terminus) and reporting whether anchors dominate;
  `map_to_positions(attr, sequence) -> list[(pos, residue, importance)]`.
- `plots.py`: `plot_residue_importance(attr, sequence, path) -> path`,
  `plot_field_importance(field_scores: dict, path) -> path`, `plot_cliff_graph(g, path, ...)`.
  Use matplotlib Agg backend; save PNG; never `plt.show()`.
- `__init__.py`: export the public functions.
Write `tests/test_interpret.py` (no torch needed): train a baseline on a few toy rows, run
`residue_attribution` over a peptide and assert it returns an array of len(peptide) with a finite
max; run `explain_baseline` and assert per-field block keys present; run `anchor_importance`.

### cli/main.py  (typer)
`app = typer.Typer(...)`; commands: `embed`, `find-cliffs`, `train`, `eval`, `explain`. Each takes
`--config PATH` (optional; default Config), common `--data`, `--output`, `--seed` overrides, calls
`seed_everything`, `setup_logging`, and the library. `def main(): app()` is the console entrypoint
(`tcr-cliff = tcr_cliff.cli.main:main`).
- `embed`: load pairs, `build_feature_matrix`, save `.npy` + names to output.
- `find-cliffs`: `find_neighbor_pairs`, write `cliffs.csv` (via `cliff_pairs_to_frame`),
  `cliff_stats.json`, and `cliff_graph.graphml`.
- `train`: load+split data, `train_model`, `model.save(output/model)`; also detect cliffs on test
  and write predictions.
- `eval`: load model + test data, `cliff_aware_report`, `save_report` + print a small table.
- `explain`: load baseline model, run `explain_baseline` + `residue_attribution` for a given row,
  save plots.
Make `cli/__init__.py` export `app`, `main`. Write `tests/test_cli.py` using Typer's
`CliRunner` (`from typer.testing import CliRunner`) to run `find-cliffs` and `train`+`eval` on the
toy data with `--config` pointing at a tmp baseline yaml (fallback embedder, small). Assert exit
code 0 and output files exist. Keep it fast (fallback_dim<=64, n_estimators<=50).

### data/download.py  + DATASETS.md
- `load_vdjdb(cache_dir=None) -> pd.DataFrame`, `load_mcpas(...)`, `load_iedb(...)`,
  `load_nettcr(...)` returning the canonical schema columns (best-effort normalisation). Each:
  document the real public source URL in the docstring, download with `urllib`/`requests` to
  `cache_dir` (default `~/.cache/tcr_cliff` or `C:\tcrtmp`), cache, parse, map columns to schema.
  On any failure raise `RuntimeError` pointing to DATASETS.md. **Do not fabricate rows.** Add
  `VDJDB_URL` etc. module constants with the canonical URLs.
- Interop helpers: `to_nettcr_format(df) -> pd.DataFrame` (NetTCR-2.0 input columns: e.g.
  `CDR3b, peptide, ...`) and `write_nettcr_csv(df, path)`; `cdr3_from_anarci(sequence, scheme="imgt")
  -> str | None` that lazy-imports `anarci`, runs IMGT numbering, extracts CDR3 (raise clear error
  if anarci missing). 
- DATASETS.md: a table citing VDJdb, McPAS-TCR, IEDB, NetTCR-2.0/ImmRep with URL, license, and
  citation, plus a clear note that the bundled toy set is **synthetic** (see
  `src/tcr_cliff/data/toy/_generate.py`) and must not be used for benchmark claims.
Write `tests/test_download.py` with `to_nettcr_format(load_toy())` (no network) asserting columns;
mark network loaders `@pytest.mark.slow` and skip by default.

### configs/  (yaml)
`default.yaml`, `baseline_lgbm.yaml`, `cliff_aware.yaml`, `fusion.yaml` — each a valid `Config`
dump (use the field names above). Baseline uses fallback embedder by default (offline-friendly)
with a commented note on switching to `esm-hf`. cliff_aware/fusion set the model kind accordingly.
Keep epochs modest. Validate each parses via `load_config`.

### README.md + .github/workflows/ci.yml + .pre-commit-config.yaml
- README.md: title + one-paragraph thesis; a **mermaid architecture diagram** (data -> embeddings
  -> cliffs -> {baseline, cliff-aware, fusion} -> cliff-aware eval -> interpret); quickstart
  (`pip install -e '.[dev]'`, `tcr-cliff find-cliffs`, `train`, `eval`); a **benchmark table
  placeholder** (models x {overall AUROC, cliff-record AUROC, gap, cliff-pair dir acc}); a
  **"Why cliffs"** explainer section; CLI reference; and a Citations section referencing
  **NetTCR-2.0 (Montemurro et al. 2021)**, **ESM-2 (Lin et al. 2023)**, and
  **AlphaFold2 (Jumper et al. 2021)** with proper references. GPLv3 note + SPDX mention.
- ci.yml: GitHub Actions, Python 3.11 & 3.12 matrix on ubuntu, `pip install -e '.[dev]'`,
  `ruff check`, `black --check`, `pytest -m 'not slow' -q`. Cache pip.
- .pre-commit-config.yaml: ruff (lint+format) and black hooks pinned to recent versions.

### cookbook/  (7 notebooks + README.md)
Valid **nbformat v4 JSON** notebooks (`nbformat=4, nbformat_minor=5`, each cell with required keys;
code cells `outputs:[], execution_count:null`). They must run top-to-bottom on the **toy data**
offline (fallback embedder) but are written so a one-line switch points at real VDJdb. Keep cells
small. Notebooks:
1. `01_load_data.ipynb` — load toy (and show how to switch to `load_vdjdb`), split, schema.
2. `02_embeddings.ipynb` — `build_feature_matrix` (fallback; note esm-hf switch), cache.
3. `03_mine_cliffs.ipynb` — `find_neighbor_pairs`, `cliff_statistics`, build + draw cliff graph,
   show a concrete single-residue flip family.
4. `04_baseline_vs_cliffaware.ipynb` — train baseline; train cliff_aware **if torch available**
   else explain; `compare_models` + show the cliff-subset performance gap (the headline).
5. `05_structure_fusion.ipynb` — load structure summary features, train fusion (if torch), measure
   lift vs baseline on cliff records.
6. `06_interpret.ipynb` — `explain_baseline` + `residue_attribution` mapped to peptide P2/PΩ
   anchors and CDR3; plot.
7. `07_usecase_bispecific_crossreactivity.ipynb` — worked use case: take a candidate target peptide,
   enumerate its single-residue neighbours, score them, and flag off-target cross-reactivity risk
   (peptides predicted to bind that are 1 edit from the target); BONUS: show `to_nettcr_format`
   export and mention ANARCI/IMGT CDR3 extraction for slotting into real pipelines.
`cookbook/README.md` indexes the notebooks and states they use the synthetic toy set offline.
Notebooks must guard heavy/torch steps with availability checks so they never hard-fail offline.
```
