import ast
import gzip
import hashlib
import math
import os
import pathlib
import random
import inspect
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from modelgenerator.structure_tokenizer.datasets.protein_dataset import ProteinDataset
from modelgenerator.structure_tokenizer.models import EquiformerEncoderLightning
from scipy.spatial.distance import cdist
from scipy.special import softmax
from tqdm.auto import tqdm

class AIDO_Structure_Tokenizer:
    def __init__(self, codebook_path=None, device="cuda", model_path=None):
        if codebook_path is None:
            codebook_path = os.environ.get("AIDO_CODEBOOK_PATH")
        if not codebook_path:
            raise ValueError(
                "AIDO structure-tokenizer codebook is not bundled in this repository. "
                "Set AIDO_CODEBOOK_PATH or pass --codebook-path."
            )

        self.codebook = torch.load(codebook_path, map_location="cpu", weights_only=True)  # [512, 384]

        # Structure encoder uses a different model: AIDO.StructureEncoder
        # This is separate from the main AIDO.Protein-16B language model
        if model_path is None:
            structure_encoder_path = "genbio-ai/AIDO.StructureEncoder"
        else:
            # If a custom path is provided, use it (but typically we use the default)
            structure_encoder_path = model_path

        self.encoder = (
            EquiformerEncoderLightning(
                pretrained_model_name_or_path=structure_encoder_path,
            ).eval().to(device)
        )

    def to(self, device):
        self.encoder = self.encoder.to(device) if self.encoder is not None else None
        return self

    def encode(self, aatype, atom_positions, atom_mask, get_embedding=False):
        """
        aatype: [L]
        atom_positions: [L, 37, 3]
        atom_mask: [L, 37]
        """
        assert self.encoder is not None, "Encoder is not loaded"

        assert aatype.ndim == 1
        assert atom_positions.ndim == 3
        assert atom_mask.ndim == 2
        device = next(iter(self.encoder.parameters())).device

        residue_index = torch.arange(1, aatype.shape[0] + 1)
        aatype = torch.from_numpy(aatype) if isinstance(aatype, np.ndarray) else aatype

        if isinstance(atom_positions, np.ndarray):
            atom_positions = atom_positions.copy()
            atom_positions[atom_mask == 0] = np.nan
            atom_mask = torch.from_numpy(atom_mask)
            atom_positions = torch.from_numpy(atom_positions).float()
        elif isinstance(atom_positions, torch.Tensor):
            atom_positions = atom_positions.clone()
            atom_positions[atom_mask == 0] = torch.nan
            atom_positions = atom_positions.float()
        else:
            raise RuntimeError(
                f"Expect atom_positions be np.ndarray or torch.Tensor, but got {type(atom_positions)}"
            )

        batch = [
            {
                "id": "xxx",
                "entity_id": "xxx",
                "chain_id": "xxx",
                "resolution": torch.tensor(3.0),
                "aatype": aatype,
                "atom_positions": atom_positions,
                "atom_mask": atom_mask,
                "residue_index": residue_index,
            }
        ]
        batch[0] = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch[0].items()}
        batch = ProteinDataset.collate_fn(batch)

        with torch.no_grad():
            tokens = self.encoder.predict_step(batch, batch_idx=0, dataloader_idx=0)["xxx_xxx_xxx"][
                "struct_tokens"
            ].cpu()

        if get_embedding:
            emb = F.embedding(tokens, self.codebook)
            return (emb, tokens)
        else:
            return tokens



def load_fasta(seqFn, rem_tVersion=False, load_annotation=False, full_line_as_id=False):
    """
    seqFn               -- Fasta file or input handle (with readline implementation)
    rem_tVersion        -- Remove version information. ENST000000022311.2 => ENST000000022311
    load_annotation     -- Load sequence annotation
    full_line_as_id     -- Use the full head line (starts with >) as sequence ID. Can not be specified simutanouly with load_annotation

    Return:
        {tid1: seq1, ...} if load_annotation==False
        {tid1: seq1, ...},{tid1: annot1, ...} if load_annotation==True
    """
    if load_annotation and full_line_as_id:
        raise RuntimeError(
            "Error: load_annotation and full_line_as_id can not be specified simutanouly"
        )
    if rem_tVersion and full_line_as_id:
        raise RuntimeError(
            "Error: rem_tVersion and full_line_as_id can not be specified simutanouly"
        )

    fasta = {}
    annotation = {}
    cur_tid = ""
    cur_seq = ""

    if isinstance(seqFn, str):
        IN = open(seqFn)
    elif hasattr(seqFn, "readline"):
        IN = seqFn
    else:
        raise RuntimeError(f"Expected seqFn: {type(seqFn)}")
    for line in IN:
        if line[0] == ">":
            if cur_seq != "":
                fasta[cur_tid] = re.sub(r"\s", "", cur_seq)
                cur_seq = ""
            data = line[1:-1].split(None, 1)
            cur_tid = line[1:-1] if full_line_as_id else data[0]
            annotation[cur_tid] = data[1] if len(data) == 2 else ""
            if rem_tVersion and "." in cur_tid:
                cur_tid = ".".join(cur_tid.split(".")[:-1])
        elif cur_tid != "":
            cur_seq += line.rstrip()

    if isinstance(seqFn, str):
        IN.close()

    if cur_seq != "":
        fasta[cur_tid] = re.sub(r"\s", "", cur_seq)

    if load_annotation:
        return fasta, annotation
    else:
        return fasta


def load_msa_a2m(file_or_stream, load_id=False, load_annot=False, sort=False):
    """
    Read MSA in A2M format (fasta-like format with gaps indicated by lowercase)
    
    Parameters
    --------------
    file_or_stream: file path or stream to read
    load_id: read identity and return
    load_annot: read annotations and return
    sort: sort by identity
    
    Return
    --------------
    msa: list of msa sequences (uppercase, gaps removed), first is query
    id_arr: Identity of msa sequences (if load_id=True)
    annotations: Annotations of msa sequences (if load_annot=True)
    """
    msa = []
    id_arr = []
    annotations = []
    
    if hasattr(file_or_stream, "read"):
        lines = file_or_stream.read().strip().split("\n")
    elif isinstance(file_or_stream, str):
        if file_or_stream.endswith(".gz"):
            with gzip.open(file_or_stream) as IN:
                lines = IN.read().decode().strip().split("\n")
        else:
            with open(file_or_stream) as IN:
                lines = IN.read().strip().split("\n")
    else:
        lines = file_or_stream
    
    current_seq = ""
    current_header = None
    q_seq = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith(">"):
            # Save previous sequence
            if current_seq:
                # Build an aligned sequence the same length as the query:
                # - Uppercase letters are kept as residues
                # - Lowercase letters (insertions) are converted to gaps ('-')
                # - Existing '-' are preserved as gaps
                seq_aligned = []
                for c in current_seq:
                    if c.isalpha():
                        if c.isupper():
                            seq_aligned.append(c)
                        else:
                            seq_aligned.append("-")
                    elif c == "-":
                        seq_aligned.append("-")
                    else:
                        seq_aligned.append("-")
                seq_aligned = "".join(seq_aligned)

                if q_seq is None:
                    q_seq = seq_aligned
                else:
                    msa.append(seq_aligned)
                    if current_header:
                        parts = current_header.split()
                        if len(parts) > 1:
                            try:
                                id_arr.append(float(parts[1]))
                            except ValueError:
                                id_ = round(np.mean([r1 == r2 for r1, r2 in zip(q_seq, seq_aligned)]), 3)
                                id_arr.append(id_)
                        else:
                            id_ = round(np.mean([r1 == r2 for r1, r2 in zip(q_seq, seq_aligned)]), 3)
                            id_arr.append(id_)
                        if len(parts) > 2:
                            annot = " ".join(parts[2:])
                            annotations.append(annot)
                        else:
                            annotations.append(None)
                    else:
                        id_ = round(np.mean([r1 == r2 for r1, r2 in zip(q_seq, seq_aligned)]), 3)
                        id_arr.append(id_)
                        annotations.append(None)
            
            current_header = line[1:]  # Remove '>'
            current_seq = ""
        else:
            current_seq += line
    
    # Save last sequence
    if current_seq:
        seq_aligned = []
        for c in current_seq:
            if c.isalpha():
                if c.isupper():
                    seq_aligned.append(c)
                else:
                    seq_aligned.append("-")
            elif c == "-":
                seq_aligned.append("-")
            else:
                seq_aligned.append("-")
        seq_aligned = "".join(seq_aligned)
        if q_seq is None:
            q_seq = seq_aligned
        else:
            msa.append(seq_aligned)
            if current_header:
                parts = current_header.split()
                if len(parts) > 1:
                    try:
                        id_arr.append(float(parts[1]))
                    except ValueError:
                        id_ = round(np.mean([r1 == r2 for r1, r2 in zip(q_seq, seq_aligned)]), 3)
                        id_arr.append(id_)
                else:
                    id_ = round(np.mean([r1 == r2 for r1, r2 in zip(q_seq, seq_aligned)]), 3)
                    id_arr.append(id_)
                if len(parts) > 2:
                    annot = " ".join(parts[2:])
                    annotations.append(annot)
                else:
                    annotations.append(None)
            else:
                id_ = round(np.mean([r1 == r2 for r1, r2 in zip(q_seq, seq_aligned)]), 3)
                id_arr.append(id_)
                annotations.append(None)
    
    id_arr = np.array(id_arr, dtype=np.float64)
    if sort:
        id_order = np.argsort(id_arr)[::-1]
        msa = [msa[i] for i in id_order]
        id_arr = id_arr[id_order]
        annotations = [annotations[i] for i in id_order]
    
    msa = [q_seq] + msa if q_seq else msa
    
    outputs = [msa]
    if load_id:
        outputs.append(id_arr)
    if load_annot:
        outputs.append(annotations)
    if len(outputs) == 1:
        return outputs[0]
    return outputs



def greedy_select(
    msa: List[Tuple[str, str]],
    num_seqs: int,
    num_tokens: int = None,
    mode: str = "max",
    seed: int = None,
) -> List[Tuple[str, str]]:
    """
    Greedy select msa sequences according to hamming distance.
    Two modes:
    - by num_seqs: select #seqs from all MSA sequences
    - by num_tokens: select a MSA sequences subset contains #tokens (excluded gap) from all MSA sequences
    """
    msa = msa.copy()
    if seed is not None:
        random.Random(seed).shuffle(msa)

    assert mode in ("max", "min")
    assert (num_seqs is None and num_tokens is not None) or (
        num_seqs is not None and num_tokens is None
    )
    if num_seqs is not None and len(msa) <= num_seqs:
        return msa
    if num_tokens is not None and sum([len(s) - s.count("-") for s in msa]) <= num_tokens:
        return msa

    array = np.array([list(seq) for seq in msa], dtype=np.bytes_).view(np.uint8)

    optfunc = np.argmax if mode == "max" else np.argmin
    all_indices = np.arange(len(msa))
    indices = [0]
    pairwise_distances = np.zeros((0, len(msa)))
    selected_msa = []
    for _ in range(len(msa) - 1):
        dist = cdist(array[indices[-1:]], array, "hamming")
        pairwise_distances = np.concatenate([pairwise_distances, dist])
        shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
        shifted_index = optfunc(shifted_distance)
        index = np.delete(all_indices, indices)[shifted_index]
        indices.append(index)
        selected_msa.append(msa[index])
        if num_seqs is not None and len(indices) >= num_seqs:
            break
        if (
            num_tokens is not None
            and sum([len(s) - s.count("-") for s in selected_msa]) >= num_tokens
        ):
            break
    indices = sorted(indices)
    return [msa[idx] for idx in indices]


def tokenize(q_seq, msa, tokenizer, max_context=12800):
    """
    Tokenizes the input sequence and optionally additional sequences for multiple sequence alignment (MSA).

    Args:
        q_seq (str): The query sequence to be tokenized.
        msa (list or None): A list of sequences for multiple sequence alignment. If None, no MSA sequences are added.
        tokenizer (object): The tokenizer object used to encode the sequences.
        max_context (int, optional): The maximum number of tokens to consider in the context. Defaults to 12800.

    Returns:
        tuple: A tuple containing:
            - tokens (np.ndarray): The tokenized sequences.
            - pos_encoding (np.ndarray): The positional encoding for the tokens.
    """
    len_seq = len(q_seq)

    def _encode(seq):
        try:
            sig = inspect.signature(tokenizer.encode)
        except (TypeError, ValueError):
            sig = None
        if sig is not None and "add_eos" in sig.parameters:
            return tokenizer.encode(seq, add_eos=False)
        return tokenizer.encode(seq)

    def _normalize_encoded(encoded_list, expected_len, label):
        if len(encoded_list) == expected_len:
            return encoded_list
        if len(encoded_list) == expected_len + 1:
            return encoded_list[1:]
        if len(encoded_list) == expected_len + 2:
            return encoded_list[1:-1]
        raise ValueError(
            f"Tokenizer returned {len(encoded_list)} tokens for {label}, expected {expected_len}"
        )

    # Handle both tensor-returning and list-returning tokenizers
    encoded = _encode(q_seq)
    encoded_list = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)
    tokens = _normalize_encoded(encoded_list, len_seq, "q_seq")
    num_seq = 1

    for msa_idx, msa_seq in enumerate(msa):
        assert len(msa_seq) == len_seq, f"len(msa_seq)={len(msa_seq)}, len_seq={len_seq}"
        encoded = _encode(msa_seq)
        encoded_list = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)
        tokens.extend(_normalize_encoded(encoded_list, len_seq, f"msa[{msa_idx}]"))
        num_seq += 1

    pos_encoding = np.stack(
        [np.tile(np.arange(len_seq), num_seq), np.repeat(np.arange(num_seq), len_seq)]
    )

    tokens = np.array(tokens)
    tok_mask = tokens != tokenizer.token_to_id("-")
    if tokens.shape[0] != pos_encoding.shape[1]:
        print(
            "[tokenize debug] mask/pos_encoding length mismatch",
            "tokens_len=", tokens.shape[0],
            "pos_encoding_len=", pos_encoding.shape[1],
        )
    tokens, pos_encoding = (
        tokens[tok_mask][:max_context],
        pos_encoding[..., tok_mask][..., :max_context],
    )
    return tokens, pos_encoding

@torch.no_grad()
def get_logits_table_sliding(
    q_seq,
    prot,
    msa,
    dms_df,
    model,
    tokenizer,
    str_tokenizer,
    start,
    sliding_window=768,
    sliding_step=768,
    mask_str=False,
    verbose=False,
    disable_tqdm=True,
):
    # model_type = get_model_type(model)
    # assert model_type == 'emb_model_step1'
    assert len(q_seq) == prot.aatype.shape[0], (
        f"len(q_seq)={len(q_seq)}, prot.aatype.shape[0]={prot.aatype.shape[0]}"
    )
    assert q_seq == msa[0]

    all_poses = set()
    for mutant in dms_df["mutant"].tolist():
        for sub_mutant in mutant.split(":"):
            sub_mutant = sub_mutant.strip()
            if len(sub_mutant) < 3:  # Skip empty or invalid mutation strings
                continue
            wt, idx, _mt = sub_mutant[0], int(sub_mutant[1:-1]) - start, sub_mutant[-1]
            try:
                seq_char = q_seq[idx]
            except Exception:
                print(
                    f"Warning: mutation index out of range or invalid: '{sub_mutant}' -> idx={idx}, start={start}, q_seq_len={len(q_seq)}"
                )
                continue

            if seq_char != wt:
                print(
                    f"Warning: WT residue mismatch for mutation '{sub_mutant}': expected '{wt}' but q_seq[{idx}]='{seq_char}'. Skipping this mutation."
                )
                continue

            all_poses.add(idx)

    all_poses = sorted(list(all_poses))
    # Get vocab size from tokenizer instead of model config

    # Prefer model vocab size to match logits dimension
    if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
        vocab_size = model.config.vocab_size
    elif hasattr(tokenizer, "vocab_size"):
        vocab_size = tokenizer.vocab_size
    elif hasattr(tokenizer, "__len__"):
        vocab_size = len(tokenizer)
    else:
        raise RuntimeError("Unable to determine vocab size from model or tokenizer")

    logit_table = np.zeros([len(all_poses), vocab_size])
    count_table = np.zeros([len(all_poses)], dtype=np.int64)

    is_last_step = False
    for f_start in range(0, len(q_seq), sliding_step):
        if is_last_step:
            break
        if f_start + sliding_window > len(q_seq) and len(q_seq) > sliding_window:
            f_start = len(q_seq) - sliding_window
            is_last_step = True

        f_end = min(f_start + sliding_window, len(q_seq))

        f_q_seq = q_seq[f_start:f_end]

        # f_msa = rag_utils.greedy_select(list(set([ seq[f_start:f_end] for seq in msa[1:] ])), num_seqs=None, num_tokens=12800, seed=0)
        f_msa = greedy_select(
            [seq[f_start:f_end] for seq in msa[1:]],
            num_seqs=None,
            num_tokens=12800,
            seed=0,
        )
        f_msa.sort(key=lambda x: x.count("-"))

        if str_tokenizer is not None and not mask_str:
            str_embs, str_toks = str_tokenizer.encode(
                prot.aatype[f_start:f_end],
                prot.atom_positions[f_start:f_end],
                prot.atom_mask[f_start:f_end],
                get_embedding=True,
            )
            str_embs, str_toks = str_embs.cuda().bfloat16(), str_toks.cuda()
        else:
            # Create zero embeddings when str_tokenizer is None or mask_str is True
            seq_len = f_end - f_start
            str_embs = torch.zeros(seq_len, 384, device='cuda', dtype=torch.bfloat16)
            str_toks = torch.zeros(seq_len, device='cuda', dtype=torch.long)

        tokens, pos_encoding = tokenize(f_q_seq, f_msa, tokenizer, max_context=12800)
        tokens = torch.from_numpy(tokens).cuda()
        pos_encoding = torch.from_numpy(pos_encoding).cuda()

        if verbose:
            tqdm.write(f"{f_start}-{f_end}, SeqL={len(q_seq)}, TokenL={tokens.shape[0]}")

        use_str_inputs = True
        def _call_model(masked_tokens, pos_encoding, str_embs):
            nonlocal use_str_inputs
            kwargs = {
                "input_ids": masked_tokens[None],
                "position_ids": pos_encoding[None],
            }
            if use_str_inputs:
                kwargs["inputs_str_embeds"] = str_embs[None]
                try:
                    return model(**kwargs)
                except TypeError:
                    use_str_inputs = False
                    kwargs.pop("inputs_str_embeds", None)
            return model(**kwargs)

        for i, pos in tqdm(
            enumerate(all_poses),
            total=len(all_poses),
            leave=False,
            dynamic_ncols=True,
            disable=disable_tqdm,
        ):
            if f_start <= pos < f_end:
                masked_tokens = tokens.clone()
                masked_tokens[pos_encoding[0] == pos - f_start] = tokenizer.token_to_id("tMASK")
                lm_output = _call_model(masked_tokens, pos_encoding, str_embs)
                if hasattr(lm_output, "logits"):
                    logits = lm_output.logits
                elif isinstance(lm_output, dict) and "logits" in lm_output:
                    logits = lm_output["logits"]
                elif isinstance(lm_output, dict) and "last_hidden_state" in lm_output and hasattr(model, "lm_head"):
                    logits = model.lm_head(lm_output["last_hidden_state"])
                else:
                    logits = lm_output[0]
                logits = logits[0, : len(q_seq)].squeeze().cpu().float()
                if logits.shape[-1] != logit_table.shape[1]:
                    print(
                        "[logits debug] vocab mismatch",
                        "logits_shape=", tuple(logits.shape),
                        "logit_table_shape=", tuple(logit_table.shape),
                        "pos=", pos,
                        "f_start=", f_start,
                        "f_end=", f_end,
                    )
                logit_table[i] += logits[pos - f_start].numpy()
                count_table[i] += 1

    if np.any(count_table == 0):
        breakpoint()
    logit_table = logit_table / count_table[:, None]
    return (all_poses, logit_table)


def get_scores_from_table(
    q_seq, logits_table, all_poses, dms_df, tokenizer, start, temp_mt=1.0, temp_wt=1.5
):
    """
    Calculate predicted scores for protein mutants based on a given table and compare them with ground truth scores.

    Args:
        q_seq (str): The query sequence of the protein.
        table (np.ndarray): A 2D array containing scores for each position and amino acid.
        all_poses (list): A list of positions in the query sequence.
        DMS_id (str): The identifier for the DMS (Deep Mutational Scanning) dataset.
        tokenizer (Tokenizer): A tokenizer object to convert amino acids to indices.
        start (int): The starting index for the positions in the query sequence.

    Returns:
        tuple: A tuple containing:
            - pd_scores (list): A list of predicted scores for each mutant.
            - gt_scores (pd.Series): A series of ground truth scores from the DMS dataset.
    """
    table_mt = np.log(softmax(logits_table / temp_mt, axis=-1))
    table_wt = np.log(softmax(logits_table / temp_wt, axis=-1))
    gt_scores = dms_df["DMS_score"]
    pd_scores = []
    vocab = tokenizer.get_vocab()
    all_data = []
    for i, mutant in enumerate(dms_df["mutant"].tolist()):
        mutant_score = 0
        for sub_mutant in mutant.split(":"):
            sub_mutant = sub_mutant.strip()
            if len(sub_mutant) < 3:  # Skip empty or invalid mutation strings
                continue
            wt, idx, mt = sub_mutant[0], int(sub_mutant[1:-1]) - start, sub_mutant[-1]
            assert wt == q_seq[idx]
            new_idx = all_poses.index(idx)
            assert new_idx >= 0
            pred = table_mt[new_idx, vocab[mt]] - table_wt[new_idx, vocab[wt]]
            mutant_score += pred.item()
        pd_scores.append(mutant_score)
        all_data.append([mutant, mutant_score, gt_scores[i]])
    result_df = pd.DataFrame(all_data, columns=["Mutation", "Pred_Score", "GT_Score"]).round(5)
    return result_df


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
    if base_cache_dir is None:
        base_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "aido")
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

def is_empty_mutation_value(value: object) -> bool:
    """Return True if a mutation value should be treated as empty/missing."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    s = str(value).strip()
    return s == "" or s.lower() in {"nan", "none", "<na>"}

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
        yield wt, pos, mt

def compute_overlap_weights(window_size: int = 1024, edge_width: int = 256) -> List[float]:
    """
    Compute the overlap weights used by the reference 'overlapping' window strategy.

    This matches the original code pattern:
        - Left decay: i in [1, 256]
        - Right decay: i in [1022-256, 1022] when window_size=1024

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

    # Left decay, indices 1..edge_width inclusive
    for i in range(1, min(edge_width, window_size - 2) + 1):
        weights[i] = 1.0 / (1.0 + math.exp(-(i - 128) / 16))

    # Right decay, matches original: i in [window_size-2-edge_width, window_size-2]
    start = max(0, (window_size - 2) - edge_width)
    for i in range(start, window_size - 1):
        # Original uses: 1 / (1 + exp((i - 1022 + 128)/16)) for window_size=1024
        anchor = window_size - 2
        weights[i] = 1.0 / (1.0 + math.exp((i - anchor + 128) / 16))

    return weights

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
        
        # Prioritize dict key order over chain_id (CSV data often has dict keys in the intended order)
        # Only use chain_id if it specifies a subset to include
        dict_keys = list(wt_dict_any.keys())
        if not chain_order:
            chain_order = dict_keys
        else:
            # If chain_id specifies chains, keep only those and preserve dict order
            chain_order_set = set(chain_order)
            chain_order = [ch for ch in dict_keys if ch in chain_order_set]
            if not chain_order:
                # Fallback to chain_id order if no dict keys match
                chain_order = [ch for ch in parse_chain_order(first_row.get(chain_id_col, "")) if ch in wt_dict_any]
                if not chain_order:
                    chain_order = dict_keys

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
        first_chain_seq = sanitize_sequence(wt_dict_any[chain_order[0]]) if chain_order and chain_order[0] in wt_dict_any else ""
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

        yield PreprocessedGroup(wt_concat=wt_concat, mutant_global=mutant_globals, row_indices=row_indices)
