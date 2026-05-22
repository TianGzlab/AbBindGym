from __future__ import annotations

import argparse
import os
from typing import List, Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer
from filelock import FileLock

# Import from IgBert's dms_utils since IgT5 is also antibody-focused
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'IgBert'))
from dms_utils import (
    build_cache_dir,
    build_output_path,
    iter_mutations,
    sanitize_sequence,
    sha256_upper,
)


class IgT5Engine:
    def __init__(
        self,
        model_path: str,
        cache_dir: str,
        *,
        device: Optional[str] = None,
        use_fp16_infer: bool = False,
    ) -> None:
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.use_fp16_infer = bool(use_fp16_infer and self.device.type == "cuda")

        print(f"Loading model from: {model_path}")
        print(f"Device: {self.device}. FP16 inference: {self.use_fp16_infer}")

        # IgT5 uses T5Tokenizer with do_lower_case=False
        self.tokenizer = T5Tokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            do_lower_case=False
        )

        # IgT5 uses T5EncoderModel
        self.model = (
            T5EncoderModel.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True
            )
            .to(self.device)
            .eval()
        )

        if self.use_fp16_infer:
            self.model = self.model.half()

        self.cache_dir = cache_dir

        os.makedirs(os.path.join(cache_dir, "wt"), exist_ok=True)

    def get_log_probs(self, sequence_heavy: str, sequence_light: str) -> torch.Tensor:
        """
        Get per-residue log-probabilities for paired antibody sequences.

        Args:
            sequence_heavy: Heavy chain sequence string.
            sequence_light: Light chain sequence string.

        Returns:
            CPU tensor of shape [L, D] where L is total sequence length (heavy + light)
            and D is the embedding dimension, stored in float16 for efficiency.
        """
        seq_heavy = sanitize_sequence(sequence_heavy)
        seq_light = sanitize_sequence(sequence_light)

        # Create a unique hash for the paired sequence
        combined_seq = seq_heavy + "|" + seq_light
        seq_hash = sha256_upper(combined_seq)

        cache_path = os.path.join(self.cache_dir, "wt", f"{seq_hash}.pt")
        lock_path = cache_path + ".lock"

        with FileLock(lock_path):
            if os.path.exists(cache_path):
                return torch.load(cache_path, map_location="cpu")

            embeddings = self._compute_embeddings(seq_heavy, seq_light)

            # Cache as float16 on CPU
            embeddings_cpu = embeddings.detach().to("cpu").half()
            torch.save(embeddings_cpu, cache_path)
            return embeddings_cpu

    def _compute_embeddings(self, sequence_heavy: str, sequence_light: str) -> torch.Tensor:
        """
        Compute residue embeddings for paired antibody sequences.

        The IgT5 tokenizer expects input of the form: "V Q L ... S S </s> E V V ... I K"
        with space-separated amino acids and </s> separator between heavy and light chains.

        Args:
            sequence_heavy: Heavy chain sequence.
            sequence_light: Light chain sequence.

        Returns:
            Tensor of shape [L, D] where L is the combined sequence length.
        """
        # Format sequences as expected by IgT5: space-separated with </s> separator
        paired_sequence = ' '.join(sequence_heavy) + ' </s> ' + ' '.join(sequence_light)

        # Tokenize
        tokens = self.tokenizer.batch_encode_plus(
            [paired_sequence],
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
            return_special_tokens_mask=True
        )

        # Move to device
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)

        # Get embeddings from encoder
        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            residue_embeddings = output.last_hidden_state[0]  # [T, D]

        # Mask out special tokens (as per IgT5 documentation)
        special_tokens_mask = tokens["special_tokens_mask"][0].to(self.device)
        residue_embeddings = residue_embeddings[special_tokens_mask == 0]  # [L, D]

        return residue_embeddings


def compute_pseudolikelihood_score(
    embeddings_wt: torch.Tensor,
    embeddings_mut: torch.Tensor,
) -> float:
    """
    Compute a pseudolikelihood-style score for mutations.

    Since IgT5 is an encoder-only model (not a masked LM), we compute scores
    based on the change in embedding similarity/distance.

    Args:
        embeddings_wt: WT embeddings [L, D].
        embeddings_mut: Mutant embeddings [L, D].

    Returns:
        A score (higher is better for the mutant).
    """
    # Simple scoring: negative L2 distance between embeddings
    # This is a proxy for "how different" the mutant is from WT
    # You may want to use cosine similarity or other metrics

    # Convert to float32 for computation
    emb_wt = embeddings_wt.float()
    emb_mut = embeddings_mut.float()

    # Compute mean squared difference
    mse = torch.mean((emb_wt - emb_mut) ** 2).item()

    # Return negative MSE (so lower distance = higher score = "more similar to WT")
    # This might need to be inverted depending on your fitness definition
    return -mse


def score_mutations(
    global_mut_heavy: str,
    global_mut_light: str,
    *,
    sequence_heavy: str,
    sequence_light: str,
    engine: IgT5Engine,
    offset_idx: int = 1,
    strict_wt_check: bool = True,
    input_csv: Optional[str] = None,
    context: Optional[str] = None
) -> float:
    """
    Score mutations by comparing WT and mutant embeddings.

    Args:
        global_mut_heavy: Mutation string for heavy chain, e.g. "H91Y:K120A".
        global_mut_light: Mutation string for light chain.
        sequence_heavy: WT heavy chain sequence.
        sequence_light: WT light chain sequence.
        engine: IgT5Engine instance.
        offset_idx: Position offset (usually 1).
        strict_wt_check: Whether to verify WT residues match.
        input_csv: Optional input file name for error messages.
        context: Optional context string for error messages.

    Returns:
        A float score.
    """
    # If both mutations are empty or "WT", return 0
    if (not global_mut_heavy or str(global_mut_heavy).strip() in {"WT", ""}) and \
       (not global_mut_light or str(global_mut_light).strip() in {"WT", ""}):
        return 0.0

    # Get WT embeddings
    embeddings_wt = engine.get_log_probs(sequence_heavy, sequence_light).float()

    # Apply mutations to create mutant sequences
    mut_heavy = list(sequence_heavy)
    mut_light = list(sequence_light)

    # Apply heavy chain mutations
    if global_mut_heavy and str(global_mut_heavy).strip() not in {"WT", ""}:
        for wt, pos, mt in iter_mutations(str(global_mut_heavy)):
            idx0 = pos - offset_idx
            if 0 <= idx0 < len(sequence_heavy):
                if strict_wt_check and sequence_heavy[idx0].upper() != wt.upper():
                    prefix = f"[{context}] " if context else ""
                    raise AssertionError(
                        f"[Input file: {input_csv or 'unknown'}] "
                        f"{prefix}WT mismatch in heavy chain at position {pos}. "
                        f"Expected {wt}, found {sequence_heavy[idx0]}"
                    )
                mut_heavy[idx0] = mt

    # Apply light chain mutations
    if global_mut_light and str(global_mut_light).strip() not in {"WT", ""}:
        for wt, pos, mt in iter_mutations(str(global_mut_light)):
            idx0 = pos - offset_idx
            if 0 <= idx0 < len(sequence_light):
                if strict_wt_check and sequence_light[idx0].upper() != wt.upper():
                    prefix = f"[{context}] " if context else ""
                    raise AssertionError(
                        f"[Input file: {input_csv or 'unknown'}] "
                        f"{prefix}WT mismatch in light chain at position {pos}. "
                        f"Expected {wt}, found {sequence_light[idx0]}"
                    )
                mut_light[idx0] = mt

    # Get mutant embeddings
    mut_heavy_seq = ''.join(mut_heavy)
    mut_light_seq = ''.join(mut_light)
    embeddings_mut = engine.get_log_probs(mut_heavy_seq, mut_light_seq).float()

    # Compute score
    score = compute_pseudolikelihood_score(
        embeddings_wt,
        embeddings_mut
    )

    return float(score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="Exscientia/IgT5",
        help="Path or HuggingFace model ID for IgT5",
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Input CSV/TSV file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for result CSVs",
    )
    parser.add_argument(
        "--focus",
        type=int,
        default=1,
        help="1=drop silent chains, 0=keep all chains",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=1,
        help="Mutation position offset, usually 1",
    )
    parser.add_argument(
        "--cache-dir",
        default="./logits_cache",
        help="Cache directory",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device string, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--fp16-infer",
        action="store_true",
        help="Enable FP16 inference on GPU",
    )
    parser.add_argument(
        "--no-strict-wt-check",
        dest="strict_wt_check",
        action="store_false",
        help="Disable WT mismatch checks",
    )
    parser.set_defaults(strict_wt_check=True)
    args = parser.parse_args()

    cache_dir = build_cache_dir(
        args.cache_dir,
        args.model_path,
        fp16=args.fp16_infer,
        focus=(args.focus == 1),
    )
    print(f"Cache dir: {cache_dir}")

    output_path = build_output_path(
        args.output_dir,
        args.input_csv,
        args.model_path,
        mode="wt",  # IgT5 only has one mode (encoder-based)
        fp16=args.fp16_infer,
        focus=(args.focus == 1),
    )
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output file: {output_path}")

    engine = IgT5Engine(
        args.model_path,
        cache_dir,
        device=args.device,
        use_fp16_infer=args.fp16_infer,
    )

    sep = "\t" if args.input_csv.endswith(".tsv") else ","
    print(f"Reading input file: {args.input_csv}")
    df = pd.read_csv(args.input_csv, sep=sep)

    # IgT5 expects paired antibody data with heavy and light chains
    # We need to extract these from the wildtype_sequence and mutant columns

    final_scores: List[Optional[float]] = [None] * len(df)

    print(f"Starting scoring with IgT5")

    # Process each row
    for idx in tqdm(range(len(df)), desc="Scoring mutations"):
        try:
            # Get wildtype sequences (assuming dict format: {'H': 'seq1', 'L': 'seq2'})
            wt_seq_dict = eval(df.loc[idx, 'wildtype_sequence']) if isinstance(df.loc[idx, 'wildtype_sequence'], str) else df.loc[idx, 'wildtype_sequence']

            # Get mutations (assuming dict format: {'H': 'H91Y:K120A', 'L': ''})
            mutant_dict = eval(df.loc[idx, 'mutant']) if isinstance(df.loc[idx, 'mutant'], str) else df.loc[idx, 'mutant']

            # Extract heavy and light chain sequences
            # Common chain IDs: H (heavy), L (light) or VH, VL
            heavy_chain_id = None
            light_chain_id = None

            for chain_id in wt_seq_dict.keys():
                if chain_id in ['H', 'VH', 'HEAVY']:
                    heavy_chain_id = chain_id
                elif chain_id in ['L', 'VL', 'LIGHT']:
                    light_chain_id = chain_id

            if heavy_chain_id is None or light_chain_id is None:
                # If we can't identify heavy/light chains, skip
                final_scores[idx] = 0.0
                continue

            seq_heavy = sanitize_sequence(wt_seq_dict[heavy_chain_id])
            seq_light = sanitize_sequence(wt_seq_dict[light_chain_id])

            mut_heavy = mutant_dict.get(heavy_chain_id, "")
            mut_light = mutant_dict.get(light_chain_id, "")

            if not seq_heavy or not seq_light:
                final_scores[idx] = 0.0
                continue

            score = score_mutations(
                mut_heavy,
                mut_light,
                sequence_heavy=seq_heavy,
                sequence_light=seq_light,
                engine=engine,
                offset_idx=args.offset,
                strict_wt_check=args.strict_wt_check,
                input_csv=args.input_csv,
                context=f"row {idx}"
            )

            final_scores[idx] = score

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            final_scores[idx] = None

    df["igt5_score"] = final_scores
    df.to_csv(output_path, index=False)
    print(f"Done. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
