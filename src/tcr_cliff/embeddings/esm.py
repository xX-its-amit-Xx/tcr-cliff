# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""ESM-2 protein language-model embedders.

Two backends are supported, both importing their heavy dependencies *lazily* so
that the dependency-free :mod:`tcr_cliff.embeddings.fallback` path keeps working
on a minimal install:

* ``esm-fair`` - ESM-2 via the original `fair-esm <https://github.com/facebookresearch/esm>`_
  package. Models are loaded through ``esm.pretrained.<model_name>()`` and batched
  with the alphabet's ``BatchConverter``.
* ``esm-hf`` - ESM-2 via HuggingFace ``transformers`` (``AutoTokenizer`` + ``EsmModel``).
  fair-esm style model names (e.g. ``esm2_t12_35M_UR50D``) are mapped to their HF
  repository ids (``facebook/esm2_t12_35M_UR50D``); fully-qualified HF ids are also
  accepted as-is.

Mean pooling averages residue representations while **excluding the BOS/EOS (and
padding) tokens**, which matters for short CDR3/peptide sequences. Per-residue
pooling returns one ``[L_i, d]`` array per sequence (no special tokens) for the
attribution code that maps importance back onto anchor positions.
"""

from __future__ import annotations

import numpy as np

from tcr_cliff._logging import get_logger
from tcr_cliff.embeddings.base import Embedder

_log = get_logger("embeddings.esm")

_ESM_EXTRA_HINT = (
    "ESM-2 backends require optional dependencies. Install them with "
    "\"pip install 'tcr-cliff[esm]'\" (this pulls in torch and either fair-esm "
    "or transformers depending on the chosen backend)."
)


def _to_hf_id(model_name: str) -> str:
    """Map a fair-esm style model name to a HuggingFace repository id.

    Parameters
    ----------
    model_name:
        Either a fair-esm name (``esm2_t12_35M_UR50D``) or an already
        fully-qualified HF id (``facebook/esm2_t12_35M_UR50D``).

    Returns
    -------
    str
        The HuggingFace repository id. Names that already contain a ``/`` are
        returned unchanged.
    """
    if "/" in model_name:
        return model_name
    return f"facebook/{model_name}"


class ESMEmbedder(Embedder):
    """ESM-2 sequence embedder with ``esm-fair`` and ``esm-hf`` backends.

    Parameters
    ----------
    model_name:
        fair-esm style model name (e.g. ``"esm2_t12_35M_UR50D"``). For the
        ``esm-hf`` backend this is mapped to a HuggingFace id when it lacks a
        ``/`` (already-qualified ids are used verbatim).
    backend:
        ``"esm-hf"`` (transformers, default) or ``"esm-fair"`` (fair-esm).
    device:
        Torch device string (``"cpu"`` default). ``"cuda"`` is honoured only when
        CUDA is actually available, otherwise the embedder falls back to CPU.
    layers:
        Transformer layer indices whose representations are averaged together.
        Negative indices count from the last layer; ``(-1,)`` (default) uses the
        final layer.
    batch_size:
        Number of sequences encoded per forward pass.
    max_len:
        Sequences longer than this (in residues) are truncated before encoding.

    Notes
    -----
    The model is loaded lazily on the first embedding call so that merely
    constructing the object (and importing this module) never imports torch.
    """

    def __init__(
        self,
        model_name: str = "esm2_t12_35M_UR50D",
        backend: str = "esm-hf",
        device: str = "cpu",
        layers: tuple[int, ...] = (-1,),
        batch_size: int = 8,
        max_len: int = 64,
    ) -> None:
        if backend not in ("esm-fair", "esm-hf"):
            raise ValueError(f"unknown ESM backend {backend!r}; expected 'esm-fair' or 'esm-hf'")
        self.model_name = str(model_name)
        self.backend = str(backend)
        self.device = str(device)
        self.layers: tuple[int, ...] = tuple(int(layer) for layer in layers)
        self.batch_size = int(batch_size)
        self.max_len = int(max_len)
        # Filesystem-safe layer tag for the cache key (no brackets/commas/spaces).
        layer_tag = "_".join(str(layer) for layer in self.layers)
        self.name = f"esm-{self.backend}-{self.model_name}-L{layer_tag}"
        # Lazily-populated handles (kept untyped to avoid importing torch at module top).
        self._model = None
        self._tokenizer = None
        self._alphabet = None
        self._batch_converter = None
        self._resolved_device: str | None = None
        self._dim: int | None = None

    # ------------------------------------------------------------------ device
    def _resolve_device(self) -> str:
        """Return the device to use, downgrading ``cuda`` to ``cpu`` if needed."""
        if self._resolved_device is not None:
            return self._resolved_device
        import torch  # lazy

        device = self.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            _log.warning("device %r requested but CUDA is unavailable; using cpu", device)
            device = "cpu"
        self._resolved_device = device
        return device

    # ------------------------------------------------------------------ loaders
    def _ensure_loaded(self) -> None:
        """Load the model/tokeniser for the selected backend (idempotent)."""
        if self._model is not None:
            return
        if self.backend == "esm-fair":
            self._load_fair()
        else:
            self._load_hf()

    def _load_fair(self) -> None:
        try:
            import esm  # type: ignore  # lazy: optional dependency
            import torch  # lazy
        except ImportError as exc:  # pragma: no cover - exercised only when dep present
            raise ImportError(
                f"The 'esm-fair' backend needs the 'fair-esm' package. {_ESM_EXTRA_HINT}"
            ) from exc

        try:
            loader = getattr(esm.pretrained, self.model_name)
        except AttributeError as exc:
            raise ValueError(
                f"fair-esm has no pretrained model named {self.model_name!r}; "
                "see esm.pretrained for available names."
            ) from exc

        model, alphabet = loader()
        device = self._resolve_device()
        model = model.to(device).eval()
        self._model = model
        self._alphabet = alphabet
        self._batch_converter = alphabet.get_batch_converter()
        self._dim = int(model.embed_dim)
        # Number of transformer layers, for translating negative indices.
        self._n_layers = int(getattr(model, "num_layers", len(getattr(model, "layers", []))))
        with torch.no_grad():
            pass
        _log.info("loaded fair-esm model %s (dim=%d) on %s", self.model_name, self._dim, device)

    def _load_hf(self) -> None:
        try:
            import torch  # lazy
            from transformers import AutoTokenizer, EsmModel  # lazy: optional dependency
        except ImportError as exc:  # pragma: no cover - exercised only when dep present
            raise ImportError(
                f"The 'esm-hf' backend needs the 'transformers' package. {_ESM_EXTRA_HINT}"
            ) from exc

        hf_id = _to_hf_id(self.model_name)
        device = self._resolve_device()
        self._tokenizer = AutoTokenizer.from_pretrained(hf_id)
        model = EsmModel.from_pretrained(hf_id, add_pooling_layer=False)
        model = model.to(device).eval()
        self._model = model
        self._dim = int(model.config.hidden_size)
        self._n_layers = int(model.config.num_hidden_layers)
        with torch.no_grad():
            pass
        _log.info("loaded HF ESM model %s (dim=%d) on %s", hf_id, self._dim, device)

    # ------------------------------------------------------------------- dim
    @property
    def dim(self) -> int:
        """Mean-pooled embedding dimensionality (hidden size of the model)."""
        if self._dim is None:
            self._ensure_loaded()
        assert self._dim is not None
        return self._dim

    # --------------------------------------------------------------- utilities
    def _abs_layers(self) -> list[int]:
        """Translate ``self.layers`` into absolute, in-range layer indices."""
        n = self._n_layers
        resolved: list[int] = []
        for layer in self.layers:
            idx = layer if layer >= 0 else n + 1 + layer  # +1: layer 0 = embeddings
            idx = max(0, min(idx, n))
            resolved.append(idx)
        return resolved

    @staticmethod
    def _batched(items: list[str], size: int):
        """Yield successive ``size``-length chunks of ``items``."""
        for start in range(0, len(items), size):
            yield start, items[start : start + size]

    def _prep(self, seq: str) -> str:
        """Uppercase and truncate a sequence to ``max_len`` residues."""
        seq = (seq or "").upper()
        return seq[: self.max_len]

    # ------------------------------------------------------------------- fair
    def _fair_reps(self, seqs: list[str]):
        """Return per-batch token representations (averaged over ``layers``).

        Yields tuples ``(reps, lengths)`` where ``reps`` is a ``[B, T, d]`` tensor
        (including BOS/EOS columns) and ``lengths`` the residue lengths per sequence.
        """
        import torch  # lazy

        device = self._resolve_device()
        layers = self._abs_layers()
        data = [(str(i), s) for i, s in enumerate(seqs)]
        for _, batch in self._batched(data, self.batch_size):
            _, _, tokens = self._batch_converter(batch)
            tokens = tokens.to(device)
            with torch.no_grad():
                out = self._model(tokens, repr_layers=layers, return_contacts=False)
            stacked = torch.stack([out["representations"][layer] for layer in layers], dim=0)
            reps = stacked.mean(dim=0)  # [B, T, d]
            lengths = [len(s) for _, s in batch]
            yield reps, lengths

    # --------------------------------------------------------------------- hf
    def _hf_reps(self, seqs: list[str]):
        """Return per-batch hidden states (averaged over ``layers``) plus masks.

        Yields ``(hidden, attention_mask)`` where ``hidden`` is ``[B, T, d]`` and
        ``attention_mask`` is ``[B, T]`` (1 for real tokens, including special ones).
        """
        import torch  # lazy

        device = self._resolve_device()
        layers = self._abs_layers()
        for _, batch in self._batched(seqs, self.batch_size):
            enc = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_len + 2,  # +2 for CLS/EOS special tokens
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = self._model(**enc, output_hidden_states=True)
            hidden_states = out.hidden_states  # tuple of [B, T, d], len n_layers + 1
            stacked = torch.stack([hidden_states[layer] for layer in layers], dim=0)
            hidden = stacked.mean(dim=0)  # [B, T, d]
            yield hidden, enc["attention_mask"]

    # ---------------------------------------------------------------- mean API
    def _embed_mean(self, sequences: list[str]) -> np.ndarray:
        """Return a ``[N, D]`` float32 matrix of mean-pooled embeddings.

        Special tokens (BOS/EOS/CLS) and padding are excluded from the mean. Empty
        sequences map to all-zero rows.
        """
        self._ensure_loaded()
        prepped = [self._prep(s) for s in sequences]
        out = np.zeros((len(prepped), self.dim), dtype=np.float32)

        if self.backend == "esm-fair":
            row = 0
            for reps, lengths in self._fair_reps(prepped):
                reps_np = reps.detach().cpu().float().numpy()
                for b, length in enumerate(lengths):
                    if length > 0:
                        # fair-esm prepends BOS at index 0; residues span [1, length].
                        out[row] = reps_np[b, 1 : length + 1].mean(axis=0)
                    row += 1
        else:
            row = 0
            for hidden, mask in self._hf_reps(prepped):
                hidden_np = hidden.detach().cpu().float().numpy()
                mask_np = mask.detach().cpu().numpy().astype(bool)
                for b in range(hidden_np.shape[0]):
                    keep = mask_np[b].copy()
                    # Drop the leading CLS and the trailing real EOS token.
                    real = np.flatnonzero(keep)
                    if real.size >= 1:
                        keep[real[0]] = False  # CLS / BOS
                    if real.size >= 2:
                        keep[real[-1]] = False  # EOS
                    if keep.any():
                        out[row] = hidden_np[b, keep].mean(axis=0)
                    row += 1
        return out

    # --------------------------------------------------------- per-residue API
    def _embed_per_residue(self, sequences: list[str]) -> list[np.ndarray]:
        """Return per-residue ``[L_i, d]`` arrays (special tokens excluded).

        Empty sequences yield ``[0, d]`` arrays so downstream indexing stays valid.
        """
        self._ensure_loaded()
        prepped = [self._prep(s) for s in sequences]
        results: list[np.ndarray] = [None] * len(prepped)  # type: ignore[list-item]

        if self.backend == "esm-fair":
            row = 0
            for reps, lengths in self._fair_reps(prepped):
                reps_np = reps.detach().cpu().float().numpy()
                for b, length in enumerate(lengths):
                    if length > 0:
                        results[row] = reps_np[b, 1 : length + 1].astype(np.float32)
                    else:
                        results[row] = np.zeros((0, self.dim), dtype=np.float32)
                    row += 1
        else:
            row = 0
            for hidden, mask in self._hf_reps(prepped):
                hidden_np = hidden.detach().cpu().float().numpy()
                mask_np = mask.detach().cpu().numpy().astype(bool)
                for b in range(hidden_np.shape[0]):
                    keep = mask_np[b].copy()
                    real = np.flatnonzero(keep)
                    if real.size >= 1:
                        keep[real[0]] = False
                    if real.size >= 2:
                        keep[real[-1]] = False
                    if keep.any():
                        results[row] = hidden_np[b, keep].astype(np.float32)
                    else:
                        results[row] = np.zeros((0, self.dim), dtype=np.float32)
                    row += 1
        return results
