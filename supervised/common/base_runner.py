#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified training runner for special models (AbLang2, AntiBERTy, IgBert).

This script provides a complete standalone training pipeline that matches train_03.py's
output structure, including:
- CSV summary files in results_root/csv/
- Prediction files in results_root/preds/
- Scatter plots in results_root/plots/
- Model checkpoints in per-fold directories

Usage:
    python -m supervised.common.base_runner \
        --dataset-name AbCoV \
        --model-key ablang2 \
        --data-path <data.csv> \
        --target-column pkd \
        --splits-path <splits.json> \
        --results-root results/supervised/<dataset> \
        [--run-tag TIMESTAMP]

    Notes:
    - target-column 'pkd' will auto-detect and convert Affinity_Kd [nM], IC50 [ug/mL], etc.
    - target-column 'ddg' will use native AB-Bind delta-delta-G labels directly.
    - Supports the same column formats as train_03.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer

# --------------------------------------------------------------------------------------
# Configuration & Global Constants
# --------------------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_CACHE = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))


# --------------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------------


def _clean_sequence(seq: str) -> str:
    """Normalize amino-acid sequence (replace rare tokens, uppercase)."""
    seq = str(seq or "")
    seq = re.sub(r"[^A-Za-z]", "", seq.upper())
    seq = re.sub(r"[UZOB]", "X", seq)
    return seq


def _optional_clean_sequence(value: object) -> Optional[str]:
    """Normalize a possibly missing sequence and return None if absent."""
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    seq = _clean_sequence(str(value or ""))
    return seq if seq else None


def _concat_antibody(hc: Optional[str], lc: Optional[str]) -> str:
    """Concatenate normalized heavy and light chains with '|' separator."""
    return f"{hc or ''}|{lc or ''}"


def _find_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Find first matching column name (case-insensitive)."""
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in df.columns:
            return name.lower()
    return None


def _safe_name(name: str) -> str:
    """Convert name to safe filename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))


# --------------------------------------------------------------------------------------
# Data Processing
# --------------------------------------------------------------------------------------


def standardize_dataframe(
    df: pd.DataFrame,
    target_column: str,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Rename key columns to canonical names for the trainer."""
    rename_map: Dict[str, str] = {}
    target_lower = str(target_column).lower()
    metadata: Dict[str, object] = {
        "target_space": target_lower,
        "affinity_source_kind": "unknown",
        "kd_metrics_enabled": False,
    }

    # Find Antigen column
    antigen_col = _find_column(
        df,
        [
            "Antigen",
            "antigen",
            "antigen_sequence",
            "Antigen_Sequence",
            "Ag_seq",
            "Ag_sequence",
            "Ag_Seq",
        ],
    )
    if antigen_col:
        rename_map[antigen_col] = "Antigen"

    # Find Heavy Chain column
    hc_col = _find_column(
        df,
        [
            "HC",
            "heavy_chain",
            "heavy",
            "Ab_heavy_chain_seq",
            "VH_abnum",
            "Ab_H_seq",
            "VH",
        ],
    )
    if hc_col:
        rename_map[hc_col] = "HC"

    # Find Light Chain column
    lc_col = _find_column(
        df,
        [
            "LC",
            "light_chain",
            "light",
            "Ab_light_chain_seq",
            "VL_abnum",
            "Ab_L_seq",
            "VL",
        ],
    )
    if lc_col:
        rename_map[lc_col] = "LC"

    # Auto-detect and convert affinity columns (similar to train_03.py)
    # First check if KD(M) column exists
    if target_lower == 'pkd' and 'KD(M)' not in df.columns:
        candidate_columns = {
            'IC50 [ug/mL]': ('surrogate_ic50', lambda x: x * (1000 / 150000) * 1e-9),  # MW=150kDa
            'Affinity_Kd [nM]': ('molar_direct', lambda x: x * 1e-9),
            'Affinity_Kd[nM]': ('molar_direct', lambda x: x * 1e-9),
            'Affinity_Kd_nM': ('molar_direct', lambda x: x * 1e-9),  # Variant without spaces
            'Kd_nM': ('molar_direct', lambda x: x * 1e-9),
            'Kd (nM)': ('molar_direct', lambda x: x * 1e-9),
            'affinity(μg/mL)': ('surrogate_elisa', lambda x: x * 1e-6 / 150000),  # ELISA, MW=150kDa
            'affinity(ng/mL)': ('surrogate_elisa', lambda x: x * 1e-9 / 150000),  # ELISA, MW=150kDa
        }

        valid_counts = {}
        for col_name, (col_type, _) in candidate_columns.items():
            if col_name in df.columns:
                valid_count = pd.to_numeric(df[col_name], errors='coerce').notna().sum()
                valid_counts[col_name] = valid_count

        if valid_counts:
            best_col = max(valid_counts, key=valid_counts.get)
            best_count = valid_counts[best_col]

            if best_count > 0:
                col_type, conversion_func = candidate_columns[best_col]
                df['KD(M)'] = pd.to_numeric(df[best_col], errors='coerce').apply(conversion_func)
                metadata["affinity_source_kind"] = col_type
                print(f"Converted {best_col} to KD(M) ({best_count} valid values)")

    # Convert KD(M) to pKd if target is pkd
    if target_lower == 'pkd':
        if 'KD(M)' in df.columns:
            df['affinity'] = df['KD(M)'].apply(lambda x: -np.log10(x) if pd.notna(x) and x > 0 else np.nan)
            if metadata["affinity_source_kind"] == "unknown":
                metadata["affinity_source_kind"] = "molar_direct"
            metadata["kd_metrics_enabled"] = (metadata["affinity_source_kind"] == "molar_direct")
            print(f"Converted KD(M) to pKd")
        elif 'pKd' in df.columns:
            rename_map['pKd'] = 'affinity'
            metadata["affinity_source_kind"] = "molar_direct"
            metadata["kd_metrics_enabled"] = True
        else:
            raise ValueError("No KD(M) or pKd column found for pKd target")
    elif target_lower == 'ddg':
        ddg_col = _find_column(df, ["ddG", "ddG(kcal/mol)", "ddg", "DDG", "DeltaDeltaG"])
        if ddg_col:
            rename_map[ddg_col] = "affinity"
            metadata["affinity_source_kind"] = "ddg_native"
            metadata["kd_metrics_enabled"] = False
        else:
            raise ValueError("No ddG column found for ddg target")
    else:
        # Find target column (original logic)
        if target_column not in df.columns:
            alt = _find_column(df, [target_column, target_column.lower()])
            if alt:
                rename_map[alt] = "affinity"
            else:
                raise ValueError(f"Target column '{target_column}' not found in dataframe.")
        else:
            rename_map[target_column] = "affinity"

    df = df.rename(columns=rename_map)

    # Validate required columns
    if "Antigen" not in df.columns:
        raise ValueError("Normalized dataframe must contain an 'Antigen' column.")

    if "HC" not in df.columns or "LC" not in df.columns:
        # Try to rescue from concatenated antibody_sequence
        antibody_col = _find_column(df, ["antibody_sequence", "Antibody", "antibody"])
        if antibody_col:
            df[["HC", "LC"]] = (
                df[antibody_col]
                .astype(str)
                .str.split("|", n=1, expand=True)
                .rename(columns={0: "HC", 1: "LC"})
            )
        else:
            raise ValueError("Neither (HC, LC) columns nor a combined antibody column found.")

    # Clean sequences while preserving missing-chain status.
    df["HC"] = df["HC"].apply(_optional_clean_sequence)
    df["LC"] = df["LC"].apply(_optional_clean_sequence)
    df["Antibody"] = df.apply(lambda row: _concat_antibody(row["HC"], row["LC"]), axis=1)
    df["Antigen"] = df["Antigen"].apply(_optional_clean_sequence)
    df["affinity"] = pd.to_numeric(df["affinity"], errors="coerce")

    # Do not drop rows here. Split scripts already perform data cleaning, and
    # preserving indices is necessary to keep them aligned with the splits JSON.
    # df = df.dropna(subset=["affinity", "Antigen"]).reset_index(drop=True)

    return df, metadata


def load_splits(path: str) -> List[Dict[str, List[int]]]:
    """Load fold splits from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    folds = payload.get("folds")
    if not folds:
        raise ValueError(f"No folds found inside {path}")
    return folds


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------


class SequenceDataset(Dataset):
    """Dataset storing cleaned strings; encoding happens inside the model."""

    def __init__(self, df: pd.DataFrame):
        self.hc = [(seq if isinstance(seq, str) else "") for seq in df["HC"].tolist()]
        self.lc = [(seq if isinstance(seq, str) else "") for seq in df["LC"].tolist()]
        self.antigen = [(seq if isinstance(seq, str) else "") for seq in df["Antigen"].tolist()]
        self.hc_present = [bool(seq) for seq in self.hc]
        self.lc_present = [bool(seq) for seq in self.lc]
        self.labels = df["affinity"].astype(np.float32).to_numpy()
        # Store original indices for prediction saving
        self.indices = df.index.tolist() if hasattr(df, 'index') else list(range(len(df)))

    def __len__(self) -> int:
        return len(self.antigen)

    def __getitem__(self, idx: int):
        return (
            self.hc[idx],
            self.lc[idx],
            self.antigen[idx],
            self.hc_present[idx],
            self.lc_present[idx],
            torch.tensor(self.labels[idx], dtype=torch.float32),
            self.indices[idx],
        )


def sequence_collate(batch):
    """Collate function for DataLoader."""
    hc, lc, antigens, hc_present, lc_present, labels, indices = zip(*batch)
    return (
        list(hc),
        list(lc),
        list(antigens),
        torch.tensor(hc_present, dtype=torch.bool),
        torch.tensor(lc_present, dtype=torch.bool),
        torch.stack(labels),
        list(indices),
    )


# --------------------------------------------------------------------------------------
# Model Encoders (AbLang2, AntiBERTy, IgBert)
# --------------------------------------------------------------------------------------


class SwiGLU(nn.Module):
    """SwiGLU activation function."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, embed_dim = x.size()
        x = x.reshape(bsz, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor, key_padding_mask: torch.Tensor):
        q = self._shape(self.q_proj(hidden_states))
        k = self._shape(self.k_proj(hidden_states))
        v = self._shape(self.v_proj(hidden_states))
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn = torch.matmul(attn_weights, v)
        attn = attn.transpose(1, 2).contiguous()
        new_shape = attn.size()[:-2] + (self.num_heads * self.head_dim,)
        attn = attn.view(*new_shape)
        return self.out_proj(attn)


class AbLangEncoderBlock(nn.Module):
    """Transformer encoder block for AbLang."""
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            SwiGLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor):
        attn = self.self_attn(x, padding_mask)
        x = x + self.dropout(attn)
        x = self.norm1(x)
        ff = self.ff(x)
        x = x + self.dropout(ff)
        x = self.norm2(x)
        return x


class AbLang2Encoder(nn.Module):
    """Light-weight AbLang2 encoder."""

    vocab = ["<pad>", "<unk>"] + list("LAGVSE R TIDPKQNFYMHWCX".replace(" ", ""))
    token_to_id = {tok: idx for idx, tok in enumerate(vocab)}

    def __init__(self, max_length: int = 512, hidden_dim: int = 768, num_layers: int = 8, num_heads: int = 12):
        super().__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(len(self.vocab), hidden_dim, padding_idx=0)
        self.layers = nn.ModuleList(
            [AbLangEncoderBlock(hidden_dim, num_heads, dropout=0.1) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.hidden_size = hidden_dim
        self.to(DEVICE)
        for param in self.parameters():
            param.requires_grad = False

    @classmethod
    def tokenize(cls, seq: str, max_length: int) -> torch.Tensor:
        seq = _clean_sequence(seq)
        token_ids = [cls.token_to_id.get(ch, 1) for ch in seq][:max_length]
        if len(token_ids) < max_length:
            token_ids += [0] * (max_length - len(token_ids))
        return torch.tensor(token_ids, dtype=torch.long)

    def forward(self, sequences: Sequence[str], max_length: Optional[int] = None) -> torch.Tensor:
        limit = max_length or self.max_length
        tokens = torch.stack([self.tokenize(seq, limit) for seq in sequences]).to(DEVICE)
        padding_mask = tokens.eq(0)
        hidden = self.embedding(tokens)
        for layer in self.layers:
            hidden = layer(hidden, padding_mask)
        hidden = self.norm(hidden)
        valid_mask = (~padding_mask).to(hidden.dtype).unsqueeze(-1)
        return (hidden * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)

    def encode(self, sequences: Sequence[str]) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(sequences)

    def encode_branch(self, sequences: Sequence[str], max_length: Optional[int] = None) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(sequences, max_length=max_length)


class AntiBERTyEncoder(nn.Module):
    """AntiBERTy encoder wrapper."""
    def __init__(self, max_length: int = 512):
        super().__init__()
        try:
            from antiberty import AntiBERTyRunner
        except Exception as exc:
            raise ImportError(
                "AntiBERTyRunner is required. Install `antiberty` package."
            ) from exc
        self.runner = AntiBERTyRunner()
        self.max_length = max_length
        self.hidden_size = 512

    def _truncate(self, seqs: Iterable[str], max_length: Optional[int] = None) -> List[str]:
        limit = max(1, (max_length or self.max_length) - 2)
        return [_clean_sequence(seq)[:limit] for seq in seqs]

    def encode(self, sequences: Sequence[str]) -> torch.Tensor:
        trunc = self._truncate(sequences)
        with torch.no_grad():
            embeds = self.runner.embed(trunc)
        pooled = []
        for emb in embeds:
            if isinstance(emb, torch.Tensor):
                pooled.append(emb.mean(dim=0))
            else:
                pooled.append(torch.zeros(self.hidden_size))
        return torch.stack(pooled).to(DEVICE)

    def encode_branch(self, sequences: Sequence[str], max_length: Optional[int] = None) -> torch.Tensor:
        trunc = self._truncate(sequences, max_length=max_length)
        with torch.no_grad():
            embeds = self.runner.embed(trunc)
        pooled = []
        for emb in embeds:
            if isinstance(emb, torch.Tensor):
                pooled.append(emb.mean(dim=0))
            else:
                pooled.append(torch.zeros(self.hidden_size))
        return torch.stack(pooled).to(DEVICE)


class IgBertEncoder(nn.Module):
    """IgBert encoder from HuggingFace."""
    def __init__(self, cache_dir: str = HF_CACHE, max_length: int = 512):
        super().__init__()
        model_id = "Exscientia/IgBert"
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True)
            self.model = BertModel.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True)
        except Exception:
            self.tokenizer = BertTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
            self.model = BertModel.from_pretrained(model_id, cache_dir=cache_dir)
        self.model.to(DEVICE)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.max_length = max_length
        self.hidden_size = self.model.config.hidden_size

    def _tokenize_sequences(self, sequences: Sequence[str], max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        formatted = [" ".join(_clean_sequence(seq)) for seq in sequences]
        tokens = self.tokenizer(
            formatted,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=max_length or self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(DEVICE) for k, v in tokens.items()}

    def encode_branch(self, sequences: Sequence[str], max_length: Optional[int] = None) -> torch.Tensor:
        with torch.no_grad():
            tokens = self._tokenize_sequences(sequences, max_length=max_length)
            outputs = self.model(**tokens)
            hidden = outputs.last_hidden_state
            mask = tokens["attention_mask"].to(hidden.dtype).unsqueeze(-1)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def encode_antibody(self, sequences: Sequence[str]) -> torch.Tensor:
        return self.encode_branch(sequences, max_length=self.max_length)

    def encode_antigen(self, sequences: Sequence[str]) -> torch.Tensor:
        return self.encode_branch(sequences, max_length=self.max_length)


class OneHotEncoder(nn.Module):
    """One-hot encoding baseline for sequences."""

    # Standard amino acids + unknown
    vocab = list("ACDEFGHIKLMNPQRSTVWYX")
    aa_to_idx = {aa: idx for idx, aa in enumerate(vocab)}

    def __init__(self, max_length: int = 512):
        super().__init__()
        self.max_length = max_length
        self.vocab_size = len(self.vocab)
        # Use a simple linear layer to project one-hot to hidden_size
        self.hidden_size = 128  # Smaller than BERT models
        self.projection = nn.Linear(self.vocab_size, self.hidden_size)
        self.to(DEVICE)

    def _encode_sequence(self, seq: str, max_length: Optional[int] = None) -> torch.Tensor:
        """Encode a single sequence to one-hot."""
        seq = _clean_sequence(seq)[:(max_length or self.max_length)]
        if not seq:
            return torch.zeros(1, self.vocab_size)
        # Create one-hot matrix: [seq_len, vocab_size]
        one_hot = torch.zeros(len(seq), self.vocab_size)
        for i, aa in enumerate(seq):
            idx = self.aa_to_idx.get(aa, self.aa_to_idx['X'])  # Unknown -> X
            one_hot[i, idx] = 1.0
        return one_hot

    def encode_branch(self, sequences: Sequence[str], max_length: Optional[int] = None) -> torch.Tensor:
        """Encode batch of sequences."""
        embeddings = []
        for seq in sequences:
            one_hot = self._encode_sequence(seq, max_length=max_length).to(DEVICE)
            pooled = one_hot.mean(dim=0)
            hidden = self.projection(pooled)
            embeddings.append(hidden)
        return torch.stack(embeddings)

    def encode(self, sequences: Sequence[str]) -> torch.Tensor:
        return self.encode_branch(sequences, max_length=self.max_length)


class AAIndexEncoder(nn.Module):
    """AAIndex-based baseline using amino acid physicochemical properties."""

    # Selected AAIndex indices (10 commonly used properties)
    # Values normalized from AAIndex database
    aaindex_properties = {
        # Format: AA -> [hydrophobicity, volume, polarity, surface_accessibility,
        #                charge, secondary_structure_propensity, solvent_accessibility,
        #                molecular_weight, isoelectric_point, flexibility]
        'A': [0.616, 0.169, 0.395, 0.539, 0.000, 0.749, 0.698, 0.191, 0.600, 0.360],
        'C': [0.680, 0.237, 0.074, 0.433, 0.000, 0.595, 0.530, 0.306, 0.517, 0.346],
        'D': [0.028, 0.244, 0.914, 0.755, -1.000, 0.429, 0.869, 0.351, 0.283, 0.511],
        'E': [0.043, 0.312, 0.914, 0.724, -1.000, 0.490, 0.860, 0.384, 0.317, 0.497],
        'F': [1.000, 0.434, 0.037, 0.314, 0.000, 0.463, 0.515, 0.433, 0.517, 0.314],
        'G': [0.501, 0.000, 0.506, 0.714, 0.000, 0.631, 0.714, 0.000, 0.600, 0.544],
        'H': [0.165, 0.339, 0.679, 0.547, 0.500, 0.542, 0.679, 0.405, 0.767, 0.323],
        'I': [0.943, 0.356, 0.037, 0.314, 0.000, 0.595, 0.478, 0.345, 0.600, 0.462],
        'K': [0.283, 0.373, 0.79, 0.857, 1.000, 0.490, 0.885, 0.382, 0.967, 0.466],
        'L': [0.943, 0.356, 0.037, 0.314, 0.000, 0.595, 0.489, 0.345, 0.600, 0.365],
        'M': [0.738, 0.356, 0.099, 0.361, 0.000, 0.595, 0.557, 0.390, 0.567, 0.295],
        'N': [0.236, 0.254, 0.827, 0.714, 0.000, 0.463, 0.828, 0.346, 0.550, 0.463],
        'P': [0.711, 0.237, 0.321, 0.490, 0.000, 0.429, 0.640, 0.303, 0.633, 0.509],
        'Q': [0.251, 0.322, 0.827, 0.686, 0.000, 0.490, 0.819, 0.382, 0.550, 0.493],
        'R': [0.000, 0.424, 0.827, 0.857, 1.000, 0.490, 0.885, 0.456, 1.000, 0.529],
        'S': [0.359, 0.186, 0.654, 0.686, 0.000, 0.542, 0.760, 0.218, 0.550, 0.507],
        'T': [0.450, 0.237, 0.580, 0.604, 0.000, 0.542, 0.707, 0.312, 0.567, 0.507],
        'V': [0.825, 0.288, 0.037, 0.361, 0.000, 0.595, 0.515, 0.265, 0.600, 0.386],
        'W': [0.878, 0.458, 0.037, 0.267, 0.000, 0.463, 0.507, 0.534, 0.583, 0.305],
        'Y': [0.880, 0.424, 0.420, 0.396, 0.000, 0.463, 0.594, 0.476, 0.550, 0.420],
        'X': [0.500, 0.250, 0.500, 0.500, 0.000, 0.500, 0.650, 0.300, 0.600, 0.400],  # Unknown
    }

    def __init__(self, max_length: int = 512):
        super().__init__()
        self.max_length = max_length
        self.n_properties = 10
        # Use a linear layer to project AAIndex features to hidden_size
        self.hidden_size = 128
        self.projection = nn.Linear(self.n_properties, self.hidden_size)
        self.to(DEVICE)

    def _encode_sequence(self, seq: str, max_length: Optional[int] = None) -> torch.Tensor:
        """Encode a single sequence using AAIndex properties."""
        seq = _clean_sequence(seq)[:(max_length or self.max_length)]
        if not seq:
            return torch.zeros(1, self.n_properties)
        # Create feature matrix: [seq_len, n_properties]
        features = torch.zeros(len(seq), self.n_properties)
        for i, aa in enumerate(seq):
            props = self.aaindex_properties.get(aa, self.aaindex_properties['X'])
            features[i] = torch.tensor(props, dtype=torch.float32)
        return features

    def encode_branch(self, sequences: Sequence[str], max_length: Optional[int] = None) -> torch.Tensor:
        """Encode batch of sequences."""
        embeddings = []
        for seq in sequences:
            features = self._encode_sequence(seq, max_length=max_length).to(DEVICE)
            pooled = features.mean(dim=0)
            hidden = self.projection(pooled)
            embeddings.append(hidden)
        return torch.stack(embeddings)

    def encode(self, sequences: Sequence[str]) -> torch.Tensor:
        return self.encode_branch(sequences, max_length=self.max_length)


def build_encoder(model_key: str, max_length: int) -> nn.Module:
    """Build encoder based on model key."""
    key = model_key.lower()
    if key in {"ablang2", "ablang"}:
        return AbLang2Encoder(max_length=max_length)
    if key in {"antiberty", "antiberty_runner"}:
        return AntiBERTyEncoder(max_length=max_length)
    if key in {"igbert"}:
        return IgBertEncoder(max_length=max_length)
    if key in {"onehot", "one-hot", "one_hot"}:
        return OneHotEncoder(max_length=max_length)
    if key in {"aaindex", "aa-index", "aa_index"}:
        return AAIndexEncoder(max_length=max_length)
    raise ValueError(f"Unsupported model key: {model_key}")


# --------------------------------------------------------------------------------------
# Affinity Regression Model
# --------------------------------------------------------------------------------------


class AffinityRegressor(nn.Module):
    """Regression head for affinity prediction."""
    def __init__(self, encoder: nn.Module, max_length: int = 512, dropout: float = 0.5):
        super().__init__()
        self.encoder = encoder
        if hasattr(encoder, "hidden_size"):
            self.hidden = encoder.hidden_size
        elif hasattr(encoder, "config"):
            self.hidden = encoder.config.hidden_size
        else:
            raise ValueError("Encoder must expose hidden_size.")
        self.max_length_ab = max_length
        self.max_length_hc_lc = max(1, max_length // 2)
        self.max_length_ag = max_length
        self.missing_hc_embedding = nn.Parameter(torch.empty(self.hidden))
        self.missing_lc_embedding = nn.Parameter(torch.empty(self.hidden))
        nn.init.normal_(self.missing_hc_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.missing_lc_embedding, mean=0.0, std=0.02)
        self.regressor = nn.Sequential(
            nn.Linear(self.hidden * 3, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def encode_branch(self, sequences: Sequence[str], max_length: int) -> torch.Tensor:
        if hasattr(self.encoder, "encode_branch"):
            return self.encoder.encode_branch(sequences, max_length=max_length)
        if hasattr(self.encoder, "encode"):
            return self.encoder.encode(sequences)
        raise AttributeError("Encoder must provide encode_branch() or encode().")

    def _apply_missing_chain_embedding(
        self,
        chain_vec: torch.Tensor,
        chain_present: torch.Tensor,
        missing_embedding: torch.Tensor,
    ) -> torch.Tensor:
        present = chain_present.to(device=chain_vec.device, dtype=torch.bool).reshape(-1, 1)
        missing_vec = missing_embedding.to(device=chain_vec.device, dtype=chain_vec.dtype).unsqueeze(0).expand(chain_vec.size(0), -1)
        return torch.where(present, chain_vec, missing_vec)

    @staticmethod
    def _materialize_missing_sequences(sequences: Sequence[str], present_flags: torch.Tensor) -> List[str]:
        return [seq if bool(present) else "X" for seq, present in zip(sequences, present_flags.tolist())]

    def forward(
        self,
        hc_sequences: Sequence[str],
        lc_sequences: Sequence[str],
        antigens: Sequence[str],
        hc_present: torch.Tensor,
        lc_present: torch.Tensor,
    ) -> torch.Tensor:
        hc_input = self._materialize_missing_sequences(hc_sequences, hc_present)
        lc_input = self._materialize_missing_sequences(lc_sequences, lc_present)
        hc_feat = self.encode_branch(hc_input, max_length=self.max_length_hc_lc)
        lc_feat = self.encode_branch(lc_input, max_length=self.max_length_hc_lc)
        ag_feat = self.encode_branch(antigens, max_length=self.max_length_ag)
        hc_feat = self._apply_missing_chain_embedding(hc_feat, hc_present, self.missing_hc_embedding)
        lc_feat = self._apply_missing_chain_embedding(lc_feat, lc_present, self.missing_lc_embedding)
        feats = torch.cat([hc_feat, lc_feat, ag_feat], dim=1)
        return self.regressor(feats).squeeze(-1)


# --------------------------------------------------------------------------------------
# Training & Evaluation
# --------------------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    """Configuration for training."""
    dataset_name: str
    model_key: str
    data_path: str
    target_column: str
    splits_path: str
    results_root: Path
    run_tag: Optional[str] = None
    epochs: int = 200
    patience: int = 30
    lr: float = 1e-4
    batch_size: int = 32
    max_length: int = 512
    clip_grad: float = 0.0
    save_preds: bool = True
    save_plots: bool = True
    save_checkpoints: bool = True


def _concordance_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Concordance correlation coefficient (Lin's CCC)."""
    x = np.asarray(y_true).astype(float)
    y = np.asarray(y_pred).astype(float)
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x), np.var(y)
    # Pearson r
    if x.size < 2:
        return np.nan
    r_num = np.sum((x - mx) * (y - my))
    r_den = np.sqrt(np.sum((x - mx) ** 2) * np.sum((y - my) ** 2))
    if r_den == 0:
        return np.nan
    r = r_num / r_den
    # CCC
    ccc = (2 * r * np.sqrt(vx) * np.sqrt(vy)) / (vx + vy + (mx - my) ** 2)
    return float(ccc)


def label_to_kd(label: np.ndarray, target: str = 'pkd') -> np.ndarray:
    """Convert pKd label back to KD(M)."""
    if target.lower() != 'pkd':
        raise ValueError(f"Unsupported KD conversion target '{target}'.")
    return 10 ** (-label)


def evaluate_with_predictions(
    model: AffinityRegressor,
    loader: DataLoader,
    target: str = 'pkd',
    using_dms: bool = False,
    kd_metrics_enabled: bool = False,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Evaluate model and return metrics + predictions (train.py compatible)."""
    model.eval()
    preds = []
    trues = []
    indices = []

    with torch.no_grad():
        for hc, lc, antigens, hc_present, lc_present, labels, batch_indices in loader:
            out = model(hc, lc, antigens, hc_present, lc_present)
            # Check for NaN in model output
            if torch.isnan(out).any():
                print(f"[WARN] Model output contains NaN values ({torch.isnan(out).sum().item()} / {out.numel()})")
                # Replace NaN with 0 to avoid breaking the evaluation
                out = torch.nan_to_num(out, nan=0.0)
            preds.append(out.cpu())
            trues.append(labels.cpu())
            indices.extend(batch_indices)

    if not preds:
        return {
            "MSE": math.nan,
            "RMSE": math.nan,
            "MAE": math.nan,
            "R2": math.nan,
            "Spearman": math.nan,
            "Spearman_p": math.nan,
            "Pearson": math.nan,
            "Pearson_p": math.nan,
            "KendallTau": math.nan,
            "KendallTau_p": math.nan,
            "CCC": math.nan,
            "GMFE": math.nan,
            "P2_within": math.nan,
            "P3_within": math.nan,
        }, pd.DataFrame()

    pred = torch.cat(preds).numpy()
    true = torch.cat(trues).numpy()

    # Check for NaN values in predictions or targets
    if np.isnan(pred).any():
        print(f"[WARN] Predictions contain {np.isnan(pred).sum()} NaN values, replacing with 0")
        pred = np.nan_to_num(pred, nan=0.0)
    if np.isnan(true).any():
        print(f"[WARN] True values contain {np.isnan(true).sum()} NaN values, replacing with 0")
        true = np.nan_to_num(true, nan=0.0)

    if scaler is not None:
        try:
            true = scaler.inverse_transform(true.reshape(-1, 1)).ravel()
            pred = scaler.inverse_transform(pred.reshape(-1, 1)).ravel()
        except Exception as exc:
            print(f"[WARN] Failed to inverse-transform predictions: {exc}")

    # Calculate all metrics (matching train.py)
    mse = float(mean_squared_error(true, pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(true, pred))

    try:
        r2 = float(r2_score(true, pred))
    except Exception:
        r2 = math.nan

    try:
        spear, spear_p = spearmanr(true, pred)
        spear = float(spear) if not math.isnan(spear) else math.nan
        spear_p = float(spear_p) if spear_p is not None else math.nan
    except Exception:
        spear = math.nan
        spear_p = math.nan

    try:
        pear, pear_p = pearsonr(true, pred)
        pear = float(pear) if not math.isnan(pear) else math.nan
        pear_p = float(pear_p) if pear_p is not None else math.nan
    except Exception:
        pear = math.nan
        pear_p = math.nan

    try:
        kendall, kendall_p = kendalltau(true, pred)
        kendall = float(kendall) if not math.isnan(kendall) else math.nan
        kendall_p = float(kendall_p) if kendall_p is not None else math.nan
    except Exception:
        kendall = math.nan
        kendall_p = math.nan

    # Calculate CCC (Concordance Correlation Coefficient)
    try:
        ccc = _concordance_ccc(true, pred)
    except Exception:
        ccc = math.nan

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Spearman": spear,
        "Spearman_p": spear_p,
        "Pearson": pear,
        "Pearson_p": pear_p,
        "KendallTau": kendall,
        "KendallTau_p": kendall_p,
        "CCC": float(ccc) if not np.isnan(ccc) else math.nan,
        "GMFE": math.nan,
        "P2_within": math.nan,
        "P3_within": math.nan,
    }

    # Calculate GMFE, P2_within, P3_within only for directly measured KD-like targets.
    if kd_metrics_enabled and not using_dms:
        try:
            kd_true = label_to_kd(np.asarray(true), target)
            kd_pred = label_to_kd(np.asarray(pred), target)
            eps = 1e-12
            ratio = np.maximum(kd_true, kd_pred) / np.maximum(np.minimum(kd_true, kd_pred), eps)
            gmfe = float(np.exp(np.mean(np.abs(np.log(np.maximum(kd_pred, eps) / np.maximum(kd_true, eps))))))
            p2 = float(np.mean(ratio <= 2.0))
            p3 = float(np.mean(ratio <= 3.0))
            metrics['GMFE'] = gmfe
            metrics['P2_within'] = p2
            metrics['P3_within'] = p3
        except Exception as e:
            print(f"[WARN] Failed to calculate GMFE/P2/P3: {e}")

    # Create predictions DataFrame
    pred_df = pd.DataFrame({
        'index': indices,
        'true': true,
        'pred': pred,
    })

    return metrics, pred_df


def save_checkpoint(
    model: AffinityRegressor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    path: Path,
) -> None:
    """Save checkpoint (matching train_03.py format)."""
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_test_loss': best_val,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model: AffinityRegressor,
    optimizer: torch.optim.Optimizer,
    path: Path,
) -> Tuple[int, float]:
    """Load checkpoint and return (epoch, best_val)."""
    if not path.exists():
        return 0, math.inf

    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint.get('epoch', 0)
    best_val = checkpoint.get('best_test_loss', math.inf)
    return epoch, best_val


def train_fold(
    cfg: TrainerConfig,
    fold_idx: int,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str = 'pkd',
    using_dms: bool = False,
    kd_metrics_enabled: bool = False,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Train a single fold and return metrics + predictions."""
    print(f"[{cfg.dataset_name}][{cfg.model_key}] Starting Fold {fold_idx}...")

    fold_scaler = StandardScaler().fit(train_df[["affinity"]])

    def _scale_df(df: pd.DataFrame) -> pd.DataFrame:
        scaled = df.copy()
        scaled["affinity"] = fold_scaler.transform(scaled[["affinity"]])
        return scaled

    train_scaled = _scale_df(train_df)
    valid_scaled = _scale_df(valid_df)
    test_scaled = _scale_df(test_df)

    encoder = build_encoder(cfg.model_key, cfg.max_length)
    model = AffinityRegressor(encoder=encoder, max_length=cfg.max_length).to(DEVICE)

    train_loader = DataLoader(
        SequenceDataset(train_scaled),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=sequence_collate,
    )
    valid_loader = DataLoader(
        SequenceDataset(valid_scaled),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=sequence_collate,
    )
    test_loader = DataLoader(
        SequenceDataset(test_scaled),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=sequence_collate,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=cfg.lr)
    best_val = math.inf
    best_state = None
    best_epoch = 0
    epochs_without_improve = 0

    # Setup checkpoint path
    checkpoint_dir = cfg.results_root / 'checkpoints' / _safe_name(cfg.model_key) / f'fold_{fold_idx}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / 'best.pt'

    # Try to resume from checkpoint
    start_epoch = 0
    if cfg.save_checkpoints and checkpoint_path.exists():
        try:
            start_epoch, best_val = load_checkpoint(model, optimizer, checkpoint_path)
            print(f"  Resumed from epoch {start_epoch} (best_val={best_val:.4f})")
        except Exception as e:
            print(f"  [WARN] Failed to load checkpoint: {e}")
            start_epoch = 0
            best_val = math.inf

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        running = 0.0
        count = 0
        for hc, lc, antigens, hc_present, lc_present, labels, _ in train_loader:
            optimizer.zero_grad()
            preds = model(hc, lc, antigens, hc_present, lc_present)

            # Check for NaN in predictions during training
            if torch.isnan(preds).any():
                print(f"[WARN] Epoch {epoch}: Training predictions contain NaN, skipping batch")
                continue

            loss = F.mse_loss(preds, labels.to(DEVICE))

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"[WARN] Epoch {epoch}: Loss is NaN, skipping batch")
                continue

            loss.backward()

            # Gradient clipping (matching train_03.py)
            if cfg.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)

            optimizer.step()
            running += loss.item() * labels.size(0)
            count += labels.size(0)

        val_metrics, _ = evaluate_with_predictions(
            model,
            valid_loader,
            target=target,
            using_dms=using_dms,
            kd_metrics_enabled=kd_metrics_enabled,
            scaler=None,
        )
        if val_metrics["MSE"] < best_val:
            best_val = val_metrics["MSE"]
            best_state = model.state_dict()
            best_epoch = epoch
            epochs_without_improve = 0

            # Save checkpoint when improvement occurs
            if cfg.save_checkpoints:
                save_checkpoint(model, optimizer, epoch, best_val, checkpoint_path)
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= cfg.patience:
            print(f"  Early stopping at epoch {epoch+1} (best epoch: {best_epoch+1})")
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on test set
    test_metrics, test_preds = evaluate_with_predictions(
        model,
        test_loader,
        target=target,
        using_dms=using_dms,
        kd_metrics_enabled=kd_metrics_enabled,
        scaler=fold_scaler,
    )

    gmfe_str = f" GMFE={test_metrics['GMFE']:.4f}" if not math.isnan(test_metrics.get('GMFE', math.nan)) else ""
    p2_str = f" P2={test_metrics['P2_within']:.4f}" if not math.isnan(test_metrics.get('P2_within', math.nan)) else ""
    p3_str = f" P3={test_metrics['P3_within']:.4f}" if not math.isnan(test_metrics.get('P3_within', math.nan)) else ""
    print(f"[{cfg.dataset_name}][{cfg.model_key}] Fold {fold_idx}: "
          f"MSE={test_metrics['MSE']:.4f} RMSE={test_metrics['RMSE']:.4f} "
          f"Spearman={test_metrics['Spearman']:.4f} Pearson={test_metrics['Pearson']:.4f}"
          f"{gmfe_str}{p2_str}{p3_str}")

    return test_metrics, test_preds


def save_summary_csv(
    cfg: TrainerConfig,
    fold_results: List[Dict[str, float]],
) -> tuple[pd.DataFrame, str]:
    """Save results to CSV summary file (train_03.py compatible format)."""
    csv_dir = cfg.results_root / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Determine dataset name from data path
    try:
        dataset_name = _safe_name(Path(cfg.data_path).stem)
    except Exception:
        dataset_name = cfg.dataset_name

    csv_path = csv_dir / f'{dataset_name}_model_summary.csv'

    # Prepare records
    records = []
    for fold_idx, metrics in enumerate(fold_results):
        row = {
            'Model': cfg.model_key,
            'Net': cfg.model_key,  # For special models, model and net are the same
            'Fold': fold_idx + 1,
            'RunTag': cfg.run_tag or datetime.now().strftime('%Y%m%d_%H%M%S'),
        }
        row.update(metrics)
        records.append(row)

    df_results = pd.DataFrame(records)

    # Merge with existing CSV if it exists
    if csv_path.exists():
        try:
            df_prev = pd.read_csv(csv_path)
            df_all = pd.concat([df_prev, df_results], ignore_index=True)
            if 'Fold' in df_all.columns:
                df_all['Fold'] = pd.to_numeric(df_all['Fold'], errors='coerce').fillna(0).astype(int)
            # Remove duplicates, keep latest
            keys = [c for c in ['Model', 'Net', 'Fold'] if c in df_all.columns]
            if keys:
                df_all = df_all.drop_duplicates(subset=keys, keep='last')
        except Exception as e:
            print(f"[WARN] Could not merge with existing CSV: {e}")
            df_all = df_results
    else:
        df_all = df_results

    df_all.to_csv(str(csv_path), index=False)
    print(f"OK: Results saved to {csv_path}")
    return df_all, dataset_name


def save_ranking_csv(
    cfg: TrainerConfig,
    df_all: pd.DataFrame,
    dataset_name: str,
) -> None:
    if df_all.empty or 'Spearman' not in df_all.columns:
        return

    csv_dir = cfg.results_root / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = csv_dir / f'{dataset_name}_ranking_by_spearman.csv'

    ranking_df = (
        df_all.groupby(['Model', 'Net'])['Spearman']
        .mean()
        .reset_index()
        .rename(columns={'Spearman': 'Spearman_mean'})
        .sort_values('Spearman_mean', ascending=False)
    )

    if ranking_path.exists():
        try:
            prev = pd.read_csv(ranking_path)
            ranking_df = pd.concat([prev, ranking_df], ignore_index=True)
            ranking_df = ranking_df.drop_duplicates(subset=['Model', 'Net'], keep='last')
            ranking_df = ranking_df.sort_values('Spearman_mean', ascending=False)
        except Exception as exc:
            print(f"[WARN] Could not merge ranking CSV: {exc}")

    ranking_df.to_csv(ranking_path, index=False)
    print(f"OK: Ranking by Spearman saved to {ranking_path}")


def save_predictions(
    cfg: TrainerConfig,
    fold_idx: int,
    predictions: pd.DataFrame,
) -> None:
    """Save per-fold predictions (train_03.py compatible format)."""
    if predictions.empty:
        return

    preds_dir = cfg.results_root / 'preds' / _safe_name(cfg.model_key)
    preds_dir.mkdir(parents=True, exist_ok=True)

    pred_path = preds_dir / f'fold{fold_idx}.csv'
    predictions.to_csv(pred_path, index=False)
    print(f"  Predictions saved to {pred_path}")


def save_scatter_plot(
    cfg: TrainerConfig,
    fold_idx: int,
    predictions: pd.DataFrame,
) -> None:
    """Save scatter plot (train_03.py compatible format)."""
    if predictions.empty:
        return

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots")
        return

    plots_dir = cfg.results_root / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    true_vals = predictions['true'].values
    pred_vals = predictions['pred'].values

    # Scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(true_vals, pred_vals, alpha=0.5, s=20)

    # Add diagonal line
    min_val = min(true_vals.min(), pred_vals.min())
    max_val = max(true_vals.max(), pred_vals.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    unit_map = {
        'pkd': 'pKd',
        'ddg': 'ddG (kcal/mol)',
        'dms_score': 'DMS score',
    }
    axis_label = unit_map.get(str(cfg.target_column).lower(), 'Affinity')
    ax.set_xlabel(f'True {axis_label}', fontsize=12)
    ax.set_ylabel(f'Predicted {axis_label}', fontsize=12)
    ax.set_title(f'{cfg.model_key} - Fold {fold_idx}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    safe_model = _safe_name(cfg.model_key)
    plot_path = plots_dir / f'{safe_model}_fold{fold_idx}_scatter.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Scatter plot saved to {plot_path}")

    # Residual diagnostics
    resid = pred_vals - true_vals
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(resid, bins=30, color='#4C78A8', alpha=0.8)
    axes[0].set_title('Residual histogram')
    axes[0].set_xlabel('Pred - True')
    axes[0].grid(axis='y', linestyle=':', alpha=0.4)

    axes[1].scatter(pred_vals, resid, s=10, alpha=0.6, color='#F58518')
    axes[1].axhline(0.0, color='r', linestyle='--', lw=1)
    axes[1].set_title('Residual vs Pred')
    axes[1].set_xlabel('Predicted value')
    axes[1].set_ylabel('Residual')
    axes[1].grid(axis='y', linestyle=':', alpha=0.4)
    fig.tight_layout()
    residual_path = plots_dir / f'{safe_model}_fold{fold_idx}_residuals.png'
    fig.savefig(residual_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Residual plot saved to {residual_path}")


def save_metrics_plot(
    cfg: TrainerConfig,
    df_all: pd.DataFrame,
) -> None:
    if df_all.empty or 'Fold' not in df_all.columns:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[WARN] matplotlib/seaborn not available, skipping metrics plot")
        return

    plots_dir = cfg.results_root / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(cfg.model_key)
    sns.barplot(x='Fold', y='MSE', data=df_all, ax=axes[0], color="#4C78A8")
    axes[0].set_title('MSE by Fold')
    axes[0].grid(axis='y', linestyle=':', alpha=0.4)

    sns.barplot(x='Fold', y='Spearman', data=df_all, ax=axes[1], color="#54A24B")
    axes[1].set_title('Spearman by Fold')
    axes[1].set_ylim(-1.0, 1.0)
    axes[1].grid(axis='y', linestyle=':', alpha=0.4)

    if 'MAE' in df_all.columns:
        sns.barplot(x='Fold', y='MAE', data=df_all, ax=axes[2], color="#F58518")
        axes[2].set_title('MAE by Fold')
        axes[2].grid(axis='y', linestyle=':', alpha=0.4)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = plots_dir / f'{_safe_name(cfg.model_key)}_metrics.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"OK: Metrics plot saved to {plot_path}")


# --------------------------------------------------------------------------------------
# Main Training Function
# --------------------------------------------------------------------------------------


def run_training(cfg: TrainerConfig):
    """Main training loop."""
    print(f"\n{'='*80}")
    print(f"Training {cfg.model_key} on {cfg.dataset_name}")
    print(f"{'='*80}")
    print(f"Data: {cfg.data_path}")
    print(f"Splits: {cfg.splits_path}")
    print(f"Results: {cfg.results_root}")
    print(f"Device: {DEVICE}")
    print(f"{'='*80}\n")

    # Load data
    df_raw = pd.read_csv(cfg.data_path)
    dataset, prep_meta = standardize_dataframe(df_raw, target_column=cfg.target_column)

    # Load splits with meta sanity checks (avoid silent index/order mismatches)
    with open(cfg.splits_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    splits = payload.get("folds") or []
    meta = payload.get("meta") or {}
    meta_size = meta.get("size") or meta.get("n")
    if meta_size is not None:
        try:
            meta_size_int = int(meta_size)
        except Exception:
            meta_size_int = None
        if meta_size_int is not None and meta_size_int != len(dataset):
            raise ValueError(
                f"splits meta.size={meta_size_int} but loaded dataset has {len(dataset)} rows. "
                f"Use the exact same CSV (and row order) that was used to generate {cfg.splits_path}."
            )
    if not splits:
        raise ValueError(f"No folds found inside {cfg.splits_path}")

    print(f"Loaded {len(dataset)} samples, {len(splits)} folds\n")

    # Detect if using DMS affinity (for BindingGYM)
    using_dms = False
    if 'DMS_score' in df_raw.columns:
        _dms_numeric = pd.to_numeric(df_raw['DMS_score'], errors='coerce')
        _has_valid_dms = _dms_numeric.notna().sum() > 0
        _path_hint = "bindinggym" in str(cfg.data_path).lower()
        using_dms = bool(_has_valid_dms and _path_hint)
        if using_dms:
            print("[INFO] Detected BindingGYM / DMS_score dataset; GMFE/P2/P3 will be set to NaN\n")
    kd_metrics_enabled = bool(prep_meta.get("kd_metrics_enabled", False))
    if str(cfg.target_column).lower() == "ddg":
        print("[INFO] Using native ddG labels; KD-based metrics will be set to NaN\n")

    fold_results: List[Dict[str, float]] = []

    for fold_idx, fold in enumerate(splits):
        train_idx = fold.get("train_idx", [])
        valid_idx = fold.get("valid_idx", [])
        test_idx = fold.get("test_idx", [])

        if not train_idx or not test_idx:
            raise ValueError(f"Fold {fold_idx} missing train/test indices.")
        max_idx = max(train_idx + valid_idx + test_idx) if (train_idx or valid_idx or test_idx) else -1
        if max_idx >= len(dataset):
            raise ValueError(
                f"Fold {fold_idx} has index {max_idx} but dataset length is {len(dataset)}. "
                "This usually means your splits JSON and data CSV do not match."
            )

        train_df = dataset.iloc[train_idx].reset_index(drop=True)
        valid_df = (
            dataset.iloc[valid_idx].reset_index(drop=True)
            if valid_idx
            else dataset.iloc[train_idx].sample(frac=0.1, random_state=314).reset_index(drop=True)
        )
        test_df = dataset.iloc[test_idx].reset_index(drop=True)

        # Train fold
        metrics, predictions = train_fold(
            cfg,
            fold_idx,
            train_df,
            valid_df,
            test_df,
            target=cfg.target_column,
            using_dms=using_dms,
            kd_metrics_enabled=kd_metrics_enabled,
        )
        fold_results.append(metrics)

        # Save predictions
        if cfg.save_preds:
            save_predictions(cfg, fold_idx, predictions)

        # Save plots
        if cfg.save_plots:
            save_scatter_plot(cfg, fold_idx, predictions)

    # Save summary CSV and aligned aggregate artifacts
    df_all, dataset_name = save_summary_csv(cfg, fold_results)
    save_ranking_csv(cfg, df_all, dataset_name)
    if cfg.save_plots:
        save_metrics_plot(cfg, df_all)

    # Print final results
    mean_spear = mean([m["Spearman"] for m in fold_results if not math.isnan(m["Spearman"])])
    mean_mse = mean([m["MSE"] for m in fold_results if not math.isnan(m["MSE"])])
    gmfe_vals = [m["GMFE"] for m in fold_results if not math.isnan(m.get("GMFE", math.nan))]
    p2_vals = [m["P2_within"] for m in fold_results if not math.isnan(m.get("P2_within", math.nan))]
    p3_vals = [m["P3_within"] for m in fold_results if not math.isnan(m.get("P3_within", math.nan))]

    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"Average MSE: {mean_mse:.4f}")
    print(f"Average Spearman: {mean_spear:.4f}")
    if gmfe_vals:
        print(f"Average GMFE: {mean(gmfe_vals):.4f}")
    if p2_vals:
        print(f"Average P2_within: {mean(p2_vals):.4f}")
    if p3_vals:
        print(f"Average P3_within: {mean(p3_vals):.4f}")
    print(f"{'='*80}\n")


# --------------------------------------------------------------------------------------
# CLI Interface
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified training runner for special models (AbLang2/AntiBERTy/IgBert)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument("--dataset-name", required=True, help="Dataset tag (e.g., AbCoV, AbDesign_ELISA)")
    parser.add_argument(
        "--model-key",
        required=True,
        choices=["ablang2", "antiberty", "igbert", "onehot", "aaindex"],
        help="Special model identifier",
    )
    parser.add_argument("--data-path", required=True, help="Path to CSV file with data")
    parser.add_argument("--target-column", required=True, help="Target type: 'pkd' for affinity-like labels, 'ddg' for native AB-Bind labels, or an explicit column name")
    parser.add_argument("--splits-path", required=True, help="Path to JSON splits file")
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Root directory for results (will create csv/, preds/, plots/ subdirs)",
    )

    # Optional arguments
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional run tag (defaults to timestamp)",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (benchmark default: 32)")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--clip-grad", type=float, default=0.0, help="Gradient clipping threshold (benchmark default: 0; set >0 to enable)")
    parser.add_argument("--no-save-preds", action="store_true", help="Disable prediction saving")
    parser.add_argument("--no-save-plots", action="store_true", help="Disable plot generation")
    parser.add_argument("--no-save-checkpoints", action="store_true", help="Disable checkpoint saving")

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Create configuration
    cfg = TrainerConfig(
        dataset_name=args.dataset_name,
        model_key=args.model_key,
        data_path=str(Path(args.data_path).expanduser().resolve()),
        target_column=args.target_column,
        splits_path=str(Path(args.splits_path).expanduser().resolve()),
        results_root=args.results_root.expanduser().resolve(),
        run_tag=args.run_tag,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        batch_size=args.batch_size,
        max_length=args.max_length,
        clip_grad=args.clip_grad,
        save_preds=not args.no_save_preds,
        save_plots=not args.no_save_plots,
        save_checkpoints=not args.no_save_checkpoints,
    )

    # Run training
    run_training(cfg)


if __name__ == "__main__":
    main()
