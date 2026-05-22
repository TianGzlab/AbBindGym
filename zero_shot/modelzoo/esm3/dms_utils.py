from __future__ import annotations

import ast
import hashlib
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd


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


def build_cache_dir(
    base_cache_dir: str, model_id: str, *, fp16: bool, focus: bool
) -> str:
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
    task_name = os.path.splitext(os.path.basename(input_csv))[0]
    model_tag = _sanitize_model_id(model_id)
    fp_tag = "fp16" if fp16 else "fp32"
    focus_tag = f"focus{int(focus)}"
    file_name = f"{task_name}__{model_tag}__{mode}__{fp_tag}__{focus_tag}.csv"
    return os.path.join(output_dir, file_name)


def sha256_upper(text: str) -> str:
    """
    Compute SHA-256 hash of an uppercased, trimmed string.

    Args:
        text: Input text.

    Returns:
        Hex digest of SHA-256.
    """
    return hashlib.sha256(str(text).strip().upper().encode("utf-8")).hexdigest()


def sanitize_sequence(seq: str) -> str:
    """
    Normalize a protein sequence string.

    Operations:
        - Strip surrounding whitespace
        - Remove internal spaces and newline characters
        - Uppercase

    Args:
        seq: Raw sequence.

    Returns:
        Sanitized sequence.
    """
    if seq is None or (isinstance(seq, float) and pd.isna(seq)):
        return ""
    return str(seq).strip().replace(" ", "").replace("\n", "").replace("\r", "").upper()


def parse_python_literal_strict(obj: object, *, field_name: str) -> object:
    """
    Parse a Python-literal string strictly via ast.literal_eval.

    This is intended for columns like:
        - wildtype_sequence: "{'A': 'SEQ...', 'B': 'SEQ...'}"
        - mutant: "{'A': 'H91Y', 'B': '', 'C': ''}"

    Args:
        obj: Value to parse.
        field_name: Used in error messages.

    Returns:
        Parsed Python object.

    Raises:
        ValueError: If parsing fails.
    """
    if obj is None:
        raise ValueError(f"{field_name} is None")
    if isinstance(obj, (dict, list, tuple)):
        return obj
    if not isinstance(obj, str):
        raise ValueError(
            f"{field_name} must be a string or a Python object, got {type(obj)}"
        )
    s = obj.strip()
    if not s:
        raise ValueError(f"{field_name} is empty")
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Failed to parse {field_name}: {e}") from e


def parse_mutant_field(obj: object, *, field_name: str) -> object:
    """
    Parse the mutant field allowing raw global mutation strings.

    This keeps strict parsing for dict-like literals but accepts plain strings like "H91Y".
    """
    if obj is None or (isinstance(obj, float) and pd.isna(obj)):
        return ""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return ""
        if s[0] in "{[(" or s[0] in {"'", '"'}:
            parsed = parse_python_literal_strict(s, field_name=field_name)
            if isinstance(parsed, (dict, str)):
                return parsed
            raise ValueError(
                f"{field_name} must be a dict or string, got {type(parsed)}"
            )
        return s
    raise ValueError(f"{field_name} must be a dict or string, got {type(obj)}")


def is_empty_mutation_value(value: object) -> bool:
    """
    Return True if a mutation value should be treated as empty/missing.
    """
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    s = str(value).strip()
    return s == "" or s.lower() in {"nan", "none", "<na>"}


def parse_chain_order(chain_id: object) -> List[str]:
    """
    Parse chain_id into an ordered list of chain identifiers.

    Supported formats:
        - "ABC" -> ["A", "B", "C"]
        - "A,B,C" -> ["A", "B", "C"]
        - "A B C" -> ["A", "B", "C"]

    Args:
        chain_id: Value from the DataFrame.

    Returns:
        Ordered list of chain IDs. Empty list if input is empty/NaN.
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


def iter_mutations(mut_str: str) -> Iterable[Tuple[str, int, str]]:
    """
    Iterate over mutations encoded as a colon-separated string.

    Example:
        "H91Y:Y92F" yields ("H", 91, "Y"), ("Y", 92, "F")

    Args:
        mut_str: Mutation string.

    Yields:
        Tuples of (wt_aa, pos_1based, mut_aa).
    """
    if mut_str is None or (isinstance(mut_str, float) and pd.isna(mut_str)):
        return
    for seg in str(mut_str).split(":"):
        m = seg.strip()
        if not m:
            continue
        # Expected format: A123B
        if len(m) < 3:
            continue
        wt = m[0].upper()
        mt = m[-1].upper()
        try:
            pos = int(m[1:-1])
        except ValueError:
            continue
        # Skip invalid positions (must be positive)
        if pos <= 0:
            continue
        yield wt, pos, mt


def compute_overlap_weights(
    window_size: int = 1024,
    edge_width: int = 256,
) -> List[float]:
    """
    Compute the overlap weights used by the reference 'overlapping' window strategy.

    This matches the original code pattern:
        - Left decay: i in [1, edge_width]
        - Right decay: i in [window_size-2-edge_width, window_size-2]

    Indices 0 (BOS/CLS) and window_size-1 (EOS) remain weight=1.

    Args:
        window_size: Window length fed into the model.
        edge_width: Width of the decayed region on each side.

    Returns:
        A list of length window_size containing float weights.
    """
    if window_size < 4:
        return [1.0] * window_size

    weights = [1.0] * window_size

    # Sigmoid parameters scaled to edge_width
    # Center at half of edge_width, steepness proportional to edge_width
    center = edge_width // 2
    steepness = edge_width / 16.0  # Scale factor for sigmoid steepness

    # Left decay, indices 1..edge_width inclusive
    for i in range(1, min(edge_width, window_size - 2) + 1):
        weights[i] = 1.0 / (1.0 + math.exp(-(i - center) / steepness))

    # Right decay, matches original: i in [window_size-2-edge_width, window_size-2]
    start = max(0, (window_size - 2) - edge_width)
    for i in range(start, window_size - 1):
        anchor = window_size - 2
        weights[i] = 1.0 / (1.0 + math.exp((i - anchor + center) / steepness))

    return weights


@dataclass(frozen=True)
class PreprocessedGroup:
    """
    A POI-level preprocessed view aligned with DMS_file_for_LLM semantics.

    Attributes:
        wt_concat: Concatenated wildtype sequence.
        mutant_global: Per-row global mutation string after chain-offset rewriting.
        row_indices: Original DataFrame indices for each row in mutant_global.
    """

    wt_concat: str
    mutant_global: List[str]
    row_indices: List[int]


def _find_focus_chains(
    g: pd.DataFrame,
    *,
    mutant_col: str,
) -> Tuple[List[str], bool]:
    """
    Replicate the focus-chains discovery from the reference DMS_file_for_LLM.

    Args:
        g: POI group DataFrame.
        mutant_col: Column name for the per-chain mutation dict.

    Returns:
        (focus_chains, has_global_mutations)
        focus_chains contains chains with at least one non-empty mutation.
        has_global_mutations is True if any row uses a global mutation string.
    """
    focus_chains: List[str] = []
    has_global_mutations = False
    for _, row in g.iterrows():
        mut_value = parse_mutant_field(row[mutant_col], field_name=mutant_col)
        if isinstance(mut_value, dict):
            for ch, m_str in mut_value.items():
                if ch not in focus_chains and not is_empty_mutation_value(m_str):
                    focus_chains.append(ch)
        else:
            if not is_empty_mutation_value(mut_value):
                has_global_mutations = True
    return focus_chains, has_global_mutations


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
    Preprocess a DMS DataFrame to match the original multi-chain concatenation logic.

    This function reproduces the key semantics of DMS_file_for_LLM:
        - Parse per-row dict strings for wildtype_sequence and mutant.
        - Determine chain order from chain_id (treated as ordered chain identifiers).
        - Concatenate chains in that order.
        - Rewrite each per-chain mutation position by adding the cumulative offset of
          previous chains in the concatenated sequence.
        - If focus=True, remove "silent" chains that never mutate in the POI group and
          recompute offsets based on the kept chains only.

    Important:
        This assumes chain_id order is consistent within each POI group. This matches
        the implicit assumption in the original scoring code.

    Args:
        df: Input DataFrame.
        wt_col: Column name of wildtype sequence dict string.
        mutant_col: Column name of mutant dict string.
        chain_id_col: Column name of chain order string.
        poi_col: Column name to group by POI. If missing, the whole df is one group.
        focus: If True, only keep chains that have at least one mutation in the group.

    Yields:
        PreprocessedGroup objects.
    """
    if poi_col in df.columns:
        groups = df.groupby(poi_col, sort=False)
    else:
        groups = [(None, df)]

    for _, g in groups:
        if len(g) == 0:
            continue

        # Chain order is taken from the first row, consistent with original usage.
        first_row = g.iloc[0]
        chain_order = parse_chain_order(first_row.get(chain_id_col, ""))

        wt_dict_any = parse_python_literal_strict(first_row[wt_col], field_name=wt_col)
        if not isinstance(wt_dict_any, dict):
            raise ValueError(
                f"{wt_col} must parse into a dict. Got {type(wt_dict_any)}"
            )

        if not chain_order:
            # Fallback to dict insertion order (least surprising if chain_id is missing).
            chain_order = list(wt_dict_any.keys())

        if focus:
            focus_chains, has_global_mutations = _find_focus_chains(
                g, mutant_col=mutant_col
            )
        else:
            focus_chains, has_global_mutations = [], False
        focus_chains_set = set(focus_chains)
        use_focus = focus and not has_global_mutations

        def keep_chain(ch: str) -> bool:
            if ch not in wt_dict_any:
                return False
            if not use_focus:
                return True
            return ch in focus_chains_set

        # Build concatenated WT and offsets
        chain_offsets: Dict[str, int] = {}
        wt_parts: List[str] = []
        offset = 0
        for ch in chain_order:
            if keep_chain(ch):
                part = sanitize_sequence(wt_dict_any[ch])
                chain_offsets[ch] = offset
                wt_parts.append(part)
                offset += len(part)

        wt_concat = "".join(wt_parts)

        # Even if wt_concat is empty, we still yield a group so the caller can set scores.
        mutant_globals: List[str] = []
        row_indices: List[int] = []

        for idx, row in g.iterrows():
            mut_value = parse_mutant_field(row[mutant_col], field_name=mutant_col)
            global_muts: List[str] = []

            if isinstance(mut_value, dict):
                for ch in chain_order:
                    if not keep_chain(ch):
                        continue
                    m_str = mut_value.get(ch, "")
                    if is_empty_mutation_value(m_str):
                        continue
                    for wt, pos, mt in iter_mutations(str(m_str)):
                        global_pos = pos + chain_offsets[ch]  # still 1-based overall
                        global_muts.append(f"{wt}{global_pos}{mt}")
            else:
                # If the column is already a global mutation string, accept it.
                if isinstance(mut_value, str) and not is_empty_mutation_value(
                    mut_value
                ):
                    global_muts.append(mut_value.strip())

            mutant_globals.append(":".join(global_muts))
            row_indices.append(idx)

        yield PreprocessedGroup(
            wt_concat=wt_concat,
            mutant_global=mutant_globals,
            row_indices=row_indices,
        )
