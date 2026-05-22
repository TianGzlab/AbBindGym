"""
Utilities for SaProt scoring: data parsing, structure handling, and model loading.

Consolidates helpers from:
- esm/dms_utils.py (sanitize, basic parsing)
- baselines/saprot/esm_loader.py (SaProt model loader)
- baselines/saprot/foldseek_util.py (structure sequence extraction)
- baselines/saprot/constants.py (vocab definitions)
"""

import ast
import hashlib
import itertools
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser

# ---- Vocab constants (from constants.py) ----
foldseek_seq_vocab = "ACDEFGHIKLMNPQRSTVWY#"
foldseek_struc_vocab = "pynwrqhgdlvtmfsaeikc#"
struc_unit = "abcdefghijklmnopqrstuvwxyz"


def sanitize_sequence(seq: str) -> str:
    if seq is None or (isinstance(seq, float) and pd.isna(seq)):
        return ""
    cleaned = str(seq).strip().replace(" ", "").replace("\n", "").replace("\r", "")
    # Check if this is a structure-aware sequence (contains lowercase letters for structure tokens)
    # If so, preserve the case; otherwise uppercase for standard amino acid sequences
    if any(c.islower() for c in cleaned):
        return cleaned  # Preserve structure-aware sequence
    return cleaned.upper()


def _sanitize_model_id(model_id: str) -> str:
    raw = str(model_id).strip()
    if not raw:
        raw = "unknown_model"
    normalized = raw
    if os.sep:
        normalized = normalized.replace(os.sep, "__")
    if os.altsep:
        normalized = normalized.replace(os.altsep, "__")
    normalized = re.sub(r"[^A-Za-z0-9._-]", "_", normalized)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"{normalized}__{digest}"


def build_cache_dir(base_cache_dir: str, model_id: str, *, fp16: bool, focus: bool) -> str:
    model_tag = _sanitize_model_id(model_id)
    fp_tag = "fp16" if fp16 else "fp32"
    focus_tag = f"focus{int(focus)}"
    return os.path.join(base_cache_dir, model_tag, fp_tag, focus_tag)


def build_output_path(
    output_dir: str,
    input_csv: str,
    model_id: str,
    *,
    mode: str,
    fp16: bool,
    focus: bool,
) -> str:
    task_name = os.path.splitext(os.path.basename(input_csv))[:1][0]
    model_tag = _sanitize_model_id(model_id)
    fp_tag = "fp16" if fp16 else "fp32"
    focus_tag = f"focus{int(focus)}"
    file_name = f"{task_name}__{model_tag}__{mode}__{fp_tag}__{focus_tag}.csv"
    return os.path.join(output_dir, file_name)


def sha256_upper(text: str) -> str:
    """Generate SHA256 hash for cache keys. Preserves case for structure-aware sequences."""
    cleaned = str(text).strip()
    # Check if this is a structure-aware sequence (contains lowercase structure tokens)
    # If so, preserve case for correct cache key differentiation
    if any(c.islower() for c in cleaned):
        return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
    return hashlib.sha256(cleaned.upper().encode("utf-8")).hexdigest()


def compute_overlap_weights(window_size: int = 1024, edge_width: int = 256) -> List[float]:
    """Compute overlap weights (ESM-style) for window aggregation."""
    if window_size < 4:
        return [1.0] * window_size

    weights = [1.0] * window_size

    # Left decay, indices 1..edge_width inclusive
    for i in range(1, min(edge_width, window_size - 2) + 1):
        weights[i] = 1.0 / (1.0 + math.exp(-(i - 128) / 16))

    # Right decay: i in [window_size-2-edge_width, window_size-2]
    start = max(0, (window_size - 2) - edge_width)
    anchor = window_size - 2
    for i in range(start, window_size - 1):
        weights[i] = 1.0 / (1.0 + math.exp((i - anchor + 128) / 16))

    return weights


def is_empty_mutation_value(value: object) -> bool:
    """Return True if a mutation value should be treated as empty/missing."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    s = str(value).strip()
    return s == "" or s.lower() in {"nan", "none", "<na>"}


def parse_chain_order(chain_id: object) -> List[str]:
    """
    Parse chain_id into an ordered list of chain identifiers.
    Supports formats: 'ABC', 'A,B,C', 'A B C', list/tuple.
    """
    if chain_id is None:
        return []
    if isinstance(chain_id, (list, tuple)):
        return [str(x).strip() for x in chain_id if str(x).strip()]
    s = str(chain_id).strip()
    if not s or s.lower() in {"nan", "none", "<na>"}:
        return []
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    if " " in s:
        return [x.strip() for x in s.split(" ") if x.strip()]
    return list(s)

    """
    Iterate over mutations encoded as a colon-separated string, yielding (wt, pos, mt).
    """
    if mut_str is None or (isinstance(mut_str, float) and pd.isna(mut_str)):
        return
    for seg in str(mut_str).split(":"):
        m = seg.strip()
        if not m or len(m) < 3:
            continue
        wt = m[0].upper()
        mt = m[-1].upper()
        try:
            pos = int(m[1:-1])
        except ValueError:
            continue
        yield wt, pos, mt


def parse_python_literal_strict(obj: object, *, field_name: str) -> object:
    """Strictly parse Python literal strings (dicts, lists, tuples)."""
    if obj is None:
        raise ValueError(f"{field_name} is None")
    if isinstance(obj, (dict, list, tuple)):
        return obj
    if not isinstance(obj, str):
        raise ValueError(f"{field_name} must be a string or a Python object, got {type(obj)}")
    s = obj.strip()
    if not s:
        raise ValueError(f"{field_name} is empty")
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Failed to parse {field_name}: {e}") from e


def parse_mutant_field(obj: object, *, field_name: str) -> object:
    """Parse mutant field allowing raw strings or dicts."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (list, tuple)):
        return obj
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return {}
        try:
            parsed = ast.literal_eval(s)
            return parsed
        except Exception:
            return s
    return obj


@dataclass(frozen=True)
class PreprocessedGroup:
    wt_concat: str
    mutant_global: List[str]
    row_indices: List[int]


def preprocess_dataframe(
    df: pd.DataFrame,
    *,
    wt_col: str = "wildtype_sequence",
    mutant_col: str = "mutant",
    chain_id_col: str = "chain_id",
    poi_col: str = "POI",
    focus: bool = True,
) -> Iterable[PreprocessedGroup]:
    """
    Simplified preprocessing to concatenate chains and rewrite mutation offsets.
    """
    if poi_col in df.columns:
        groups = df.groupby(poi_col, sort=False)
    else:
        groups = [(None, df)]

    for _, g in groups:
        if len(g) == 0:
            continue

        first_row = g.iloc[0]
        chain_order = parse_chain_order(first_row.get(chain_id_col, ""))
        wt_dict_any = parse_python_literal_strict(first_row[wt_col], field_name=wt_col)
        if not isinstance(wt_dict_any, dict):
            raise ValueError(f"{wt_col} must parse into a dict.")
        if not chain_order:
            chain_order = list(wt_dict_any.keys())

        focus_chains = []
        if focus:
            for _, row in g.iterrows():
                mut_val = parse_mutant_field(row[mutant_col], field_name=mutant_col)
                if isinstance(mut_val, dict):
                    for ch, m_str in mut_val.items():
                        if ch not in focus_chains and not is_empty_mutation_value(m_str):
                            focus_chains.append(ch)
        focus_set = set(focus_chains)

        def keep_chain(ch: str) -> bool:
            if ch not in wt_dict_any:
                return False
            if not focus:
                return True
            return ch in focus_set

        chain_offsets: Dict[str, int] = {}
        wt_parts: List[str] = []
        offset = 0

        # Check if we're working with structure-aware sequences (lowercase structure tokens present)
        first_chain_seq = (
            sanitize_sequence(wt_dict_any[chain_order[0]])
            if chain_order and chain_order[0] in wt_dict_any
            else ""
        )
        is_structure_aware = any(c.islower() for c in first_chain_seq)

        for ch in chain_order:
            if keep_chain(ch):
                part = sanitize_sequence(wt_dict_any[ch])
                chain_offsets[ch] = offset
                wt_parts.append(part)
                # For structure-aware sequences, offset is in amino acid space (each AA = 2 chars)
                # For regular sequences, offset is in character space (each AA = 1 char)
                if is_structure_aware:
                    offset += len(part) // 2
                else:
                    offset += len(part)
        wt_concat = "".join(wt_parts)

        mutant_globals: List[str] = []
        row_indices: List[int] = []
        for idx, row in g.iterrows():
            mut_value = parse_mutant_field(row[mutant_col], field_name=mutant_col)
            global_muts: List[str] = []
            if isinstance(mut_value, dict):
                for ch in chain_order:
                    if not keep_chain(ch):
                        continue
                    # Verify this chain is in our offsets (it should be if keep_chain returned True)
                    if ch not in chain_offsets:
                        print(f"Warning: chain {ch} not in chain_offsets, skipping mutations")
                        continue
                    m_str = mut_value.get(ch, "")
                    if is_empty_mutation_value(m_str):
                        continue
                    for wt, pos, mt in iter_mutations(str(m_str)):
                        global_pos = pos + chain_offsets[ch]
                        global_muts.append(f"{wt}{global_pos}{mt}")
            else:
                if isinstance(mut_value, str) and not is_empty_mutation_value(mut_value):
                    global_muts.append(mut_value.strip())
            mutant_globals.append(":".join(global_muts))
            row_indices.append(idx)

        yield PreprocessedGroup(
            wt_concat=wt_concat, mutant_global=mutant_globals, row_indices=row_indices
        )


def compute_overlap_weights(window_size: int = 1024, edge_width: int = 256) -> List[float]:
    """Compute overlap weights (ESM-style) for window aggregation."""
    if window_size < 4:
        return [1.0] * window_size

    weights = [1.0] * window_size

    # Left decay, indices 1..edge_width inclusive
    for i in range(1, min(edge_width, window_size - 2) + 1):
        weights[i] = 1.0 / (1.0 + math.exp(-(i - 128) / 16))

    # Right decay: i in [window_size-2-edge_width, window_size-2]
    start = max(0, (window_size - 2) - edge_width)
    anchor = window_size - 2
    for i in range(start, window_size - 1):
        weights[i] = 1.0 / (1.0 + math.exp((i - anchor + 128) / 16))

    return weights


def is_empty_mutation_value(value: object) -> bool:
    """Return True if a mutation value should be treated as empty/missing."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    s = str(value).strip()
    return s == "" or s.lower() in {"nan", "none", "<na>"}


def parse_chain_order(chain_id: object) -> List[str]:
    """
    Parse chain_id into an ordered list of chain identifiers.
    Supports formats: 'ABC', 'A,B,C', 'A B C', list/tuple.
    """
    if chain_id is None:
        return []
    if isinstance(chain_id, (list, tuple)):
        return [str(x).strip() for x in chain_id if str(x).strip()]
    s = str(chain_id).strip()
    if not s or s.lower() in {"nan", "none", "<na>"}:
        return []
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    if " " in s:
        return [x.strip() for x in s.split(" ") if x.strip()]
    return list(s)


def iter_mutations(mut_str: str):
    """
    Iterate over mutations encoded as a colon-separated string, yielding (wt, pos, mt).
    """
    if mut_str is None or (isinstance(mut_str, float) and pd.isna(mut_str)):
        return
    for seg in str(mut_str).split(":"):
        m = seg.strip()
        if not m or len(m) < 3:
            continue
        wt = m[0].upper()
        mt = m[-1].upper()
        try:
            pos = int(m[1:-1])
        except ValueError:
            continue
        yield wt, pos, mt


# ---- SaProt model loader (from esm_loader.py) ----
def load_weights(model, weights):
    model_dict = model.state_dict()
    unused_params = []
    missed_params = list(model_dict.keys())
    for k, v in weights.items():
        if k in model_dict.keys():
            model_dict[k] = v
            missed_params.remove(k)
        else:
            unused_params.append(k)
    if len(missed_params) > 0:
        print(
            f"Some weights of {type(model).__name__} were not initialized from the model checkpoint: {missed_params}"
        )
    if len(unused_params) > 0:
        print(f"Some weights of the model checkpoint were not used: {unused_params}")
    model.load_state_dict(model_dict)


def load_esm_saprot(path: str):
    """
    Load SaProt model of esm version.
    Args:
        path: path to SaProt model
    """
    import esm
    from esm.model.esm2 import ESM2

    # Initialize the alphabet
    tokens = ["<cls>", "<pad>", "<eos>", "<unk>", "<mask>"]
    for seq_token, struc_token in itertools.product(foldseek_seq_vocab, foldseek_struc_vocab):
        token = seq_token + struc_token
        tokens.append(token)

    alphabet = esm.data.Alphabet(
        standard_toks=tokens,
        prepend_toks=[],
        append_toks=[],
        prepend_bos=True,
        append_eos=True,
        use_msa=False,
    )

    alphabet.all_toks = alphabet.all_toks[:-2]
    alphabet.unique_no_split_tokens = alphabet.all_toks
    alphabet.tok_to_idx = {tok: i for i, tok in enumerate(alphabet.all_toks)}

    # Load weights
    data = torch.load(path)
    weights = data["model"]
    config = data["config"]

    # Initialize the model
    model = ESM2(
        num_layers=config["num_layers"],
        embed_dim=config["embed_dim"],
        attention_heads=config["attention_heads"],
        alphabet=alphabet,
        token_dropout=config["token_dropout"],
    )

    load_weights(model, weights)
    return model, alphabet


# ---- Structure handling (from foldseek_util.py) ----
biopython_pdbparser = PDBParser(QUIET=True)
alpha_3 = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "MSE",
]
alphabet = "ACDEFGHIKLMNPQRSTVWY"


def get_struc_seq(
    path,
    wt_seq_dic=None,
    python=None,
    chains: list = None,
    process_id: int = 0,
    plddt_path: str = None,
    plddt_threshold: float = 70.0,
) -> dict:
    """
    Extract structure-derived sequence annotations for the requested chains.
    Returns a dict: chain_id -> (seq, struc_seq, combined_seq)
    """
    assert os.path.exists(path), f"Pdb file not found: {path}"
    assert plddt_path is None or os.path.exists(plddt_path), f"Plddt file not found: {plddt_path}"

    tmp_dir = Path("src/saprot/tmp_save_paths")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_save_path = tmp_dir / f"get_struc_seq_{process_id}.tsv"

    cmd = f"{os.path.dirname(python)}/foldseek structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    os.system(cmd)

    s = biopython_pdbparser.get_structure("", path)

    seq_dict = {}
    name = os.path.basename(path)
    with open(tmp_save_path, "r") as r:
        for _, line in enumerate(r):
            desc, seq, struc_seq = line.split("\t")[:3]

            # Mask low pLDDT
            if plddt_path is not None:
                with open(plddt_path, "r") as r_plddt:
                    plddts = np.array(json.load(r_plddt)["confidenceScore"])
                    indices = np.where(plddts < plddt_threshold)[0]
                    np_seq = np.array(list(struc_seq))
                    np_seq[indices] = "#"
                    struc_seq = "".join(np_seq)

            name_chain = desc.split(" ")[0]
            chain = name_chain.replace(name, "").split("_")[-1]

            if chains is None or chain in chains:
                if chain not in seq_dict:
                    revise_seq = []
                    res_i = 0
                    for res in s[0][chain].get_residues():
                        if res.get_full_id()[-1][0] != " ":
                            continue
                        if res_i >= len(seq):
                            break
                        if res.resname not in alpha_3:
                            revise_seq.append("X")
                        else:
                            revise_seq.append(seq[res_i])
                        res_i += 1
                    seq = "".join(revise_seq)
                    if len(seq) != len(struc_seq):
                        min_len = min(len(seq), len(struc_seq))
                        seq = seq[:min_len]
                        struc_seq = struc_seq[:min_len]
                    wt_seq = wt_seq_dic[chain]
                    wt_i = 0
                    seq_i = 0
                    pad_seq = []
                    pad_struc_seq = []
                    while seq_i < len(seq):
                        aa = seq[seq_i]
                        if aa not in alphabet:
                            pad_seq.append("#")
                            pad_struc_seq.append(struc_seq[seq_i])
                            seq_i += 1
                        elif aa == wt_seq[wt_i] and aa in alphabet:
                            pad_seq.append(aa)
                            pad_struc_seq.append(struc_seq[seq_i])
                            wt_i += 1
                            seq_i += 1
                        elif wt_seq[wt_i] == "X":
                            pad_seq.append("#")
                            pad_struc_seq.append("#")
                            wt_i += 1
                        if wt_i >= len(wt_seq):
                            break
                    if wt_i < len(wt_seq):
                        for _ in range(wt_i, len(wt_seq)):
                            wt_i += 1
                            pad_seq.append("#")
                            pad_struc_seq.append("#")
                    seq = "".join(pad_seq)
                    struc_seq = "".join(pad_struc_seq)
                    combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                    seq_dict[chain] = (seq, struc_seq, combined_seq)

    os.remove(tmp_save_path)
    dbtype_path = tmp_save_path.with_suffix(tmp_save_path.suffix + ".dbtype")
    if dbtype_path.exists():
        os.remove(dbtype_path)
    return seq_dict


# ---- DMS processing (copied from utils/data_utils.py) ----
def DMS_file_for_LLM(df, focus=False, return_focus_chains=False):
    df["chain_id"] = df["chain_id"].fillna("")
    df["wildtype_sequence"] = df["wildtype_sequence"].apply(ast.literal_eval)
    df["mutant"] = df["mutant"].apply(ast.literal_eval)
    df["mutated_sequence"] = df["mutated_sequence"].apply(ast.literal_eval)
    input_wt_seqs = []
    input_mt_seqs = []
    input_focus_wt_seqs = []
    input_focus_mt_seqs = []
    input_mutants = []
    input_focus_mutants = []
    focus_chains = []
    for i in df.index:
        mutants = df.loc[i, "mutant"]
        for c in mutants:
            if c not in focus_chains:
                if mutants[c] != "":
                    focus_chains.append(c)
    for i in df.index:
        chain_ids = df.loc[i, "chain_id"]
        wt_seqs = ""
        mt_seqs = ""
        focus_wt_seqs = ""
        focus_mt_seqs = ""
        wt_seq_dic = df.loc[i, "wildtype_sequence"]
        mt_seq_dic = df.loc[i, "mutated_sequence"]
        mutants = df.loc[i, "mutant"]
        revise_mutants = []
        focus_revise_mutants = []
        start_idx = 0
        focus_start_idx = 0
        for _, chain_id in enumerate(chain_ids):
            ms = mutants.get(chain_id, "")
            if ms != "":
                for m in ms.split(":"):
                    pos = int(m[1:-1]) + start_idx
                    revise_mutants.append(m[:1] + str(pos) + m[-1:])
            wt_seqs += wt_seq_dic.get(chain_id, "")
            mt_seqs += mt_seq_dic.get(chain_id, "")
            start_idx += len(wt_seq_dic.get(chain_id, ""))
            if chain_id in focus_chains:
                if ms != "":
                    for m in ms.split(":"):
                        pos = int(m[1:-1]) + focus_start_idx
                        focus_revise_mutants.append(m[:1] + str(pos) + m[-1:])
                focus_wt_seqs += wt_seq_dic.get(chain_id, "")
                focus_mt_seqs += mt_seq_dic.get(chain_id, "")
                focus_start_idx += len(wt_seq_dic.get(chain_id, ""))

        input_wt_seqs.append(wt_seqs)
        input_mt_seqs.append(mt_seqs)
        input_mutants.append(":".join(revise_mutants))

        input_focus_wt_seqs.append(focus_wt_seqs)
        input_focus_mt_seqs.append(focus_mt_seqs)
        input_focus_mutants.append(":".join(focus_revise_mutants))
    if not focus:
        df["wildtype_sequence"] = input_wt_seqs
        df["mutated_sequence"] = input_mt_seqs
        df["mutant"] = input_mutants
    else:
        df["wildtype_sequence"] = input_focus_wt_seqs
        df["mutated_sequence"] = input_focus_mt_seqs
        df["mutant"] = input_focus_mutants
    if return_focus_chains:
        return df, sorted(focus_chains)
    return df
