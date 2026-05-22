from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from data_utils import DMS_file_for_LLM


def create_model(ckpt: str, fp16: bool):
    """
    Load ProGen2 model via transformers AutoModelForCausalLM from a local checkpoint directory (offline-safe).

    Note:
      - ckpt should be a resolved local directory, or a repo id that has already been cached.
      - local_files_only=True ensures no network calls happen.
    """
    model = AutoModelForCausalLM.from_pretrained(
        ckpt,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.float16 if fp16 else None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def create_tokenizer(ckpt_dir: str) -> Tokenizer:
    return Tokenizer.from_pretrained(ckpt_dir)


########################################################################
# fitness


def _chunk_sequence(seq: str, chunk_len: int) -> List[str]:
    """
    Split sequence into non-empty contiguous chunks of length <= chunk_len.
    """
    if chunk_len <= 0:
        raise ValueError(f"chunk_len must be > 0, got {chunk_len}")
    if len(seq) <= chunk_len:
        return [seq]
    chunks: List[str] = []
    for start in range(0, len(seq), chunk_len):
        end = min(start + chunk_len, len(seq))
        if end > start:
            chunks.append(seq[start:end])
    return chunks


def calc_fitness(
    model,
    prots,
    tokenizer,
    device: str = "cuda:0",
    model_context_len: int = 1024,
    fp16: bool = False,
    reduction: str = "sum",
):
    """
    Compute fitness score (negative CE loss) for each protein sequence in `prots`.

    Maintains original behavior:
      - chunking for long sequences
      - mirroring score by forward and reversed chunk, then normalize by 2
      - remove terminals if last token is BOS or EOS
      - restrict logits to AA vocab tokens [5..29], shift targets accordingly
      - output is numpy array of per-seq scores
    """
    loss_list: List[float] = []
    loss_fn = CrossEntropyLoss()

    # constants kept identical to your original implementation
    bos_token, eos_token = 3, 4
    first_token, last_token = 5, 29

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=fp16):
            for prot in tqdm(prots, desc="Scoring", leave=False):
                loss_val = 0.0

                # chunking, ensure no empty chunks
                sequence_chunks = _chunk_sequence(str(prot), model_context_len)

                for chunk in sequence_chunks:
                    # mirror forward and reverse
                    for p in (chunk, chunk[::-1]):
                        ids = torch.tensor(
                            tokenizer.encode(p).ids, device=device, dtype=torch.long
                        )

                        if ids.numel() < 2:
                            # cannot form input-target pair
                            continue

                        input_ids = ids[:-1]
                        targets = ids[1:]

                        # enforce [batch, seq] input for maximal compatibility
                        out = model(input_ids.unsqueeze(0))
                        logits = out.logits.squeeze(0)  # [L, vocab]

                        # remove terminals if last target token is BOS/EOS
                        if targets.numel() > 0 and targets[-1].item() in (
                            bos_token,
                            eos_token,
                        ):
                            logits = logits[:-1, ...]
                            targets = targets[:-1]

                        if targets.numel() == 0:
                            continue

                        # sanity checks, preserve original assertions as runtime guards
                        if (targets == bos_token).any().item():
                            raise AssertionError("Targets contain BOS token unexpectedly.")
                        if (targets == eos_token).any().item():
                            raise AssertionError("Targets contain EOS token unexpectedly.")

                        # restrict logits to AA token range and shift targets
                        logits = logits[:, first_token : (last_token + 1)]
                        targets = targets - first_token

                        if logits.shape[1] != (last_token - first_token + 1):
                            raise AssertionError("Unexpected restricted vocab size.")

                        loss = loss_fn(
                            input=logits.reshape(-1, logits.size(-1)),
                            target=targets.reshape(-1),
                        )
                        loss_val += -float(loss.item())

                # normalize for mirroring
                loss_val /= 2.0

                if reduction == "mean":
                    loss_val /= max(1, len(str(prot)))

                loss_list.append(loss_val)

    return np.array(loss_list, dtype=np.float32)


def get_mutated_sequence(
    focus_seq, mutant, start_idx=1, AA_vocab="ACDEFGHIKLMNPQRSTVWY"
):
    """
    Helper function that mutates an input sequence (focus_seq) via an input mutation triplet (substitutions only).
    Mutation triplet are typically based on 1-indexing: start_idx is used for switching to 0-indexing.
    """
    mutated_seq = list(focus_seq)
    for mutation in mutant.split(":"):
        try:
            from_AA, position, to_AA = mutation[0], int(mutation[1:-1]), mutation[-1]
        except Exception:
            print("Issue with mutant: " + str(mutation))
            continue

        relative_position = position - start_idx
        assert from_AA == focus_seq[relative_position], (
            "Invalid from_AA or mutant position: "
            + str(mutation)
            + " from_AA: "
            + str(from_AA)
            + " relative pos: "
            + str(relative_position)
            + " focus_seq: "
            + str(focus_seq)
        )
        assert to_AA in AA_vocab, "Mutant to_AA is invalid: " + str(mutation)
        mutated_seq[relative_position] = to_AA
    return "1" + "".join(mutated_seq) + "2"


def main() -> None:
    """
    Main script to score sets of mutated protein sequences (substitutions or indels) with ProGen2.
    """
    parser = argparse.ArgumentParser(description="ProGen2 scoring (transformers, offline-safe)")

    parser.add_argument(
        "--checkpoint",
        default="/n/groups/marks/projects/marks_lab_and_oatml/protein_transformer/baseline_models/progen2/progen2-small",
        type=str,
        help="HuggingFace repo id (e.g., hugohrban/progen2-small) or local path to a ProGen2 checkpoint directory",
    )
    parser.add_argument(
        "--dms_mapping",
        default=None,
        type=str,
        help="Path of DMS mapping file",
    )
    parser.add_argument(
        "--dms_input",
        default="/n/groups/marks/projects/marks_lab_and_oatml/protein_transformer/Tranception_open_source/DMS_files/ProteinGym_substitutions",
        type=str,
        help="Path of DMS folder or a single CSV file",
    )
    parser.add_argument(
        "--dms_index",
        type=int,
        help="Index in mapping file for which DMS dataset to score",
    )
    parser.add_argument(
        "--dms_output",
        default=None,
        type=str,
        help="Folder to write model scores to",
    )
    parser.add_argument(
        "--indel_mode",
        action="store_true",
        help="Whether to score sequences with insertions and deletions",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to score sequences with half precision",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode of fitness computation",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="Device to run scoring on, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--focus",
        type=int,
        default=1,
        help="1=drop silent chains, 0=keep all chains (passed to DMS_file_for_LLM)",
    )

    args = parser.parse_args()

    focus_flag = bool(args.focus == 1)

    # Resolve checkpoint to local directory (offline-safe)
    # ckpt_dir = resolve_ckpt_dir(args.checkpoint)
    ckpt_dir = args.checkpoint
    print(f"Resolved checkpoint dir: {ckpt_dir}")

    # Load model via transformers
    model = create_model(ckpt=ckpt_dir, fp16=args.fp16)
    n_positions = int(model.config.n_positions)
    tokenizer = create_tokenizer(ckpt_dir=ckpt_dir)

    # Load DMS data
    dms_input_is_csv = args.dms_input.endswith(".csv") or (
        ".csv" in os.path.basename(args.dms_input)
    )

    if not dms_input_is_csv:
        if args.dms_index is None:
            raise ValueError("--dms_index must be provided when --dms_input is a directory")

        if args.dms_mapping is None:
            raise ValueError("--dms_mapping must be provided when --dms_input is a directory")
        mapping_protein_seq_DMS = pd.read_csv(args.dms_mapping)
        if (
            "DMS_id" not in mapping_protein_seq_DMS.columns
            or "DMS_filename" not in mapping_protein_seq_DMS.columns
        ):
            raise ValueError("DMS mapping file must contain columns: DMS_id, DMS_filename")

        list_DMS = mapping_protein_seq_DMS["DMS_id"].tolist()
        if args.dms_index < 0 or args.dms_index >= len(list_DMS):
            raise IndexError(
                f"--dms_index out of range. got {args.dms_index}, total {len(list_DMS)}"
            )

        DMS_id = list_DMS[args.dms_index]
        print(f"Computing scores for: {DMS_id} with Progen2: {args.checkpoint}")

        DMS_file_name = mapping_protein_seq_DMS.loc[
            mapping_protein_seq_DMS["DMS_id"] == DMS_id, "DMS_filename"
        ].values
        if len(DMS_file_name) != 1:
            raise ValueError(f"Could not uniquely resolve DMS_filename for DMS_id: {DMS_id}")
        DMS_file_name = DMS_file_name[0]

        df = pd.read_csv(os.path.join(args.dms_input, DMS_file_name), low_memory=False)

        if args.dms_output is None:
            raise ValueError("--dms_output must be provided to write output files")
        os.makedirs(args.dms_output, exist_ok=True)
        scoring_filename = os.path.join(args.dms_output, f"{DMS_id}_focus1.csv")
    else:
        df = pd.read_csv(args.dms_input, low_memory=False)
        if args.dms_output is None:
            raise ValueError("--dms_output must be provided to write output files")
        os.makedirs(args.dms_output, exist_ok=True)
        scoring_filename = os.path.join(args.dms_output, f"{os.path.basename(args.dms_input)}_focus1.csv")

    if "POI" not in df.columns:
        raise ValueError("Input DMS csv must contain column: POI")

    # Compute scores
    all_g = []
    for POI, g in df.groupby("POI"):
        g = DMS_file_for_LLM(g, focus=focus_flag)
        print(POI)

        if "wildtype_sequence" not in g.columns or "mutated_sequence" not in g.columns:
            raise ValueError(
                "DMS_file_for_LLM output must contain columns: wildtype_sequence, mutated_sequence"
            )

        wt_seq = g["wildtype_sequence"].values[0]
        inputs = ["1" + wt_seq + "2"] + [
            "1" + s + "2" for s in g["mutated_sequence"].tolist()
        ]

        model_scores = calc_fitness(
            model=model,
            prots=np.array(inputs),
            model_context_len=n_positions,
            tokenizer=tokenizer,
            fp16=args.fp16,
            device=args.device,
        )

        # mutant score is delta relative to WT
        g["Progen2_score"] = model_scores[1:] - model_scores[0]
        all_g.append(g)

    out_df = pd.concat(all_g).sort_index()
    out_df.to_csv(scoring_filename, index=False)
    print(f"Wrote scores to: {scoring_filename}")


if __name__ == "__main__":
    main()
