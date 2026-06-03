# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Typed, validated configuration for every stage of the pipeline.

Configs are pydantic models so they validate on load and round-trip to/from YAML.
A single :class:`Config` object is threaded through ``embed``/``find-cliffs``/
``train``/``eval``/``explain`` so that runs are fully reproducible from one file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class DataConfig(BaseModel):
    """Where pairs come from and which columns to use."""

    source: Literal["toy", "csv", "vdjdb", "mcpas", "iedb", "nettcr"] = "toy"
    path: Path | None = Field(
        default=None, description="CSV path for source='csv', or a cache dir for downloaders."
    )
    test_size: float = Field(default=0.2, ge=0.0, lt=1.0)
    val_size: float = Field(default=0.1, ge=0.0, lt=1.0)
    split_column: str | None = Field(
        default=None,
        description="If set, use this column's train/val/test labels instead of a random split.",
    )
    group_split_on: Literal["none", "peptide", "cdr3b"] = Field(
        default="peptide",
        description="Grouped split key to avoid leakage (no peptide/CDR3 in both train and test).",
    )


class EmbeddingConfig(BaseModel):
    """ESM-2 / fallback embedding extraction."""

    backend: Literal["fallback", "esm-fair", "esm-hf"] = "fallback"
    model_name: str = "esm2_t12_35M_UR50D"
    pooling: Literal["mean", "per_residue"] = "mean"
    layers: list[int] = Field(default_factory=lambda: [-1])
    include_mhc: bool = True
    cache_dir: Path | None = Path(".tcr_cliff_cache/embeddings")
    device: str = "cpu"
    batch_size: int = Field(default=8, ge=1)
    max_len: int = 64
    # Fallback (numpy, dependency-free) embedder knobs.
    fallback_dim: int = Field(default=320, ge=8)
    kmer_k: int = Field(default=3, ge=1, le=4)

    @property
    def fields(self) -> tuple[str, ...]:
        """Sequence fields embedded (peptide + CDR3b, optionally MHC pseudo-seq)."""
        base = ("cdr3b", "peptide")
        return (*base, "mhc_pseudo") if self.include_mhc else base


class CliffConfig(BaseModel):
    """Definition of a TCR-pMHC *binding cliff*.

    A cliff is a pair of records whose sequences are within ``max_edits`` edits on
    the varied entity but whose binding outcome differs (label flip and/or affinity
    delta above ``affinity_delta``), while the *context* (the non-varied entity, and
    MHC) is held fixed when ``same_context`` is True.
    """

    max_edits: int = Field(default=1, ge=1, description="k: max edit distance to be 'neighbours'.")
    distance: Literal["levenshtein", "hamming"] = "levenshtein"
    vary: Literal["peptide", "cdr3b", "both"] = "both"
    require_label_flip: bool = True
    affinity_delta: float | None = Field(
        default=None, description="If set, abs affinity difference above this also marks a cliff."
    )
    same_context: bool = Field(
        default=True, description="Require the non-varied entity (and MHC) to match exactly."
    )
    min_len: int = Field(default=5, ge=1, description="Ignore sequences shorter than this.")


class StructureConfig(BaseModel):
    """AlphaFold/ESMFold confidence ingestion."""

    enabled: bool = False
    source_dir: Path | None = Field(
        default=None, description="Directory of per-pair confidence JSON/PDB files."
    )
    use_stub: bool = Field(
        default=True, description="Use bundled/precomputed confidence files (no GPU folding)."
    )
    pae_interface_cutoff: float = Field(
        default=8.0, gt=0, description="PAE (Angstrom) threshold for an inter-chain contact."
    )


class BaselineModelConfig(BaseModel):
    """ESM-embeddings + gradient-boosted baseline."""

    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    prefer: Literal["lightgbm", "sklearn"] = "lightgbm"


class CliffAwareModelConfig(BaseModel):
    """Contrastive cliff-aware pretraining + supervised head."""

    proj_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.1
    # Contrastive pretraining.
    contrastive_epochs: int = 30
    contrastive_margin: float = 1.0
    contrastive_temperature: float = 0.2
    cliff_weight: float = 2.0  # up-weight pushing cliff pairs apart
    # Supervised head.
    head_epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    freeze_encoder_after_pretrain: bool = False


class FusionModelConfig(BaseModel):
    """Concatenate structural-confidence features onto the sequence head."""

    structure_feature_dim: int = 12
    hidden_dim: int = 128
    dropout: float = 0.1


class ModelConfig(BaseModel):
    kind: Literal["baseline_lgbm", "cliff_aware", "fusion"] = "baseline_lgbm"
    baseline: BaselineModelConfig = Field(default_factory=BaselineModelConfig)
    cliff_aware: CliffAwareModelConfig = Field(default_factory=CliffAwareModelConfig)
    fusion: FusionModelConfig = Field(default_factory=FusionModelConfig)


class TrainConfig(BaseModel):
    seed: int = 0
    device: str = "cpu"
    early_stopping_patience: int = 10
    log_every: int = 5


class EvalConfig(BaseModel):
    task: Literal["classification", "regression", "both"] = "classification"
    threshold: float = 0.5
    bootstrap: int = Field(default=0, ge=0, description="Bootstrap resamples for CIs (0 = off).")
    report_cliff_subset: bool = True


class Config(BaseModel):
    """Top-level run configuration."""

    seed: int = 0
    output_dir: Path = Path("runs/default")
    data: DataConfig = Field(default_factory=DataConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    cliff: CliffConfig = Field(default_factory=CliffConfig)
    structure: StructureConfig = Field(default_factory=StructureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _sync_seed(self) -> Config:
        # A single top-level seed propagates to the train block unless overridden.
        if self.train.seed == 0 and self.seed != 0:
            self.train.seed = self.seed
        return self

    def to_yaml(self, path: str | Path) -> Path:
        """Serialise the config to a YAML file and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(_jsonable(self.model_dump()), sort_keys=False))
        return path


def _jsonable(obj: Any) -> Any:
    """Recursively convert Paths to strings for clean YAML output."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load a :class:`Config` from YAML, applying optional keyword overrides.

    Parameters
    ----------
    path:
        Path to a YAML config. If ``None``, defaults are used.
    **overrides:
        Top-level keys to override after loading (shallow merge).

    Returns
    -------
    Config
        The validated configuration.
    """
    data: dict[str, Any] = {}
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    data.update(overrides)
    return Config.model_validate(data)
