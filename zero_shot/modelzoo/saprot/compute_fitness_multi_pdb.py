from __future__ import annotations

import argparse
import ast
import os
from typing import Optional

import pandas as pd
import torch
from filelock import FileLock
from transformers import AutoModelForMaskedLM, AutoTokenizer

from data_utils import (
    build_cache_dir,
    build_output_path,
    compute_overlap_weights,
    get_struc_seq,
    preprocess_dataframe,
    sanitize_sequence,
    sha256_upper,
)

DEFAULT_FOLDSEEK_ROOT = os.environ.get("FOLDSEEK_ROOT", "bin/foldseek")


class SaProtEngine:
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

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )

        self.model = (
            AutoModelForMaskedLM.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )

        if self.use_fp16_infer:
            self.model = self.model.half()

        self.cache_dir = cache_dir
        os.makedirs(os.path.join(cache_dir, "wt"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "masked"), exist_ok=True)

        self.window_size: int = int(getattr(self.model.config, "max_position_embeddings", 1024))
        self.vocab_size: int = int(self.model.config.vocab_size)
        self.mask_token_id: int = int(self.tokenizer.mask_token_id)

        w = compute_overlap_weights(self.window_size)
        self.window_weights = torch.tensor(w, device=self.device, dtype=torch.float32)

    def get_log_probs(self, sequence: str, mode: str) -> torch.Tensor:
        print(
            f"[get_log_probs] Input sequence length: {len(sequence)}, first 50 chars: {sequence[:50]}"
        )
        seq = sanitize_sequence(sequence)
        print(f"[get_log_probs] After sanitize: length {len(seq)}, first 50 chars: {seq[:50]}")
        seq_hash = sha256_upper(seq)

        cache_path = os.path.join(self.cache_dir, mode, f"{seq_hash}.pt")
        lock_path = f"{cache_path}.lock"
        with FileLock(lock_path):
            if os.path.exists(cache_path):
                try:
                    return torch.load(cache_path, map_location="cpu")
                except Exception as exc:
                    print(f"Cache load failed at {cache_path} ({exc}); recomputing.")
                    try:
                        os.remove(cache_path)
                    except OSError:
                        pass

            if mode == "wt":
                log_probs = self._compute_wt_overlapping(seq)
            elif mode == "masked":
                log_probs = self._compute_masked_optimal_batched(seq)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            log_probs_cpu = log_probs.detach().to("cpu").half()
            torch.save(log_probs_cpu, cache_path)
            return log_probs_cpu

    def _model_forward_log_probs(self, input_ids_1d: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            logits = self.model(input_ids=input_ids_1d.unsqueeze(0)).logits[0]
            if self.use_fp16_infer:
                logits = logits.float()
            return torch.log_softmax(logits, dim=-1)

    def _compute_wt_overlapping(self, sequence: str) -> torch.Tensor:
        print(
            f"[_compute_wt_overlapping] Sequence length: {len(sequence)}, first 50: {sequence[:50]}"
        )
        inputs = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs["input_ids"][0].to(self.device)
        T = int(input_ids.size(0))
        print(
            f"[_compute_wt_overlapping] Tokenized to T={T} tokens, window_size={self.window_size}"
        )
        if T <= self.window_size:
            lprobs = self._model_forward_log_probs(input_ids)
            return lprobs[1:-1, :]

        token_accum = torch.zeros((T, self.vocab_size), device=self.device, dtype=torch.float32)
        weight_accum = torch.zeros((T,), device=self.device, dtype=torch.float32)

        stride = max(1, (self.window_size // 2) - 1)
        start_left = 0
        end_left = self.window_size - 1
        start_right = (T - 1) - self.window_size + 1
        end_right = T - 1

        while True:
            chunk_left = input_ids[start_left : end_left + 1]
            w_left = self.window_weights[: chunk_left.size(0)]
            lprobs_left = self._model_forward_log_probs(chunk_left)
            token_accum[start_left : end_left + 1] += lprobs_left * w_left.unsqueeze(-1)
            weight_accum[start_left : end_left + 1] += w_left

            chunk_right = input_ids[start_right : end_right + 1]
            w_right = self.window_weights[: chunk_right.size(0)]
            lprobs_right = self._model_forward_log_probs(chunk_right)
            token_accum[start_right : end_right + 1] += lprobs_right * w_right.unsqueeze(-1)
            weight_accum[start_right : end_right + 1] += w_right

            if end_left > start_right:
                break
            start_left += stride
            end_left += stride
            start_right -= stride
            end_right -= stride

        final_overlap = end_left - start_right + 1
        if final_overlap < stride:
            start_center = max(0, (T // 2) - (self.window_size // 2))
            end_center = min(T - 1, start_center + self.window_size - 1)
            chunk_center = input_ids[start_center : end_center + 1]
            w_center = self.window_weights[: chunk_center.size(0)]
            lprobs_center = self._model_forward_log_probs(chunk_center)
            token_accum[start_center : end_center + 1] += lprobs_center * w_center.unsqueeze(-1)
            weight_accum[start_center : end_center + 1] += w_center

        weight_accum = torch.clamp(weight_accum, min=1e-6)
        final_lprobs = token_accum / weight_accum.unsqueeze(-1)
        return final_lprobs[1:-1, :]

    def _compute_masked_optimal_batched(
        self, sequence: str, batch_size: int = 1280
    ) -> torch.Tensor:
        inputs = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        input_ids_full = inputs["input_ids"][0].to(self.device)
        T = int(input_ids_full.size(0))
        L = T - 2
        V = self.vocab_size
        W = self.window_size
        half = W // 2

        final_lprobs = torch.empty((L, V), device=self.device, dtype=torch.float32)
        token_idx_all = torch.arange(1, L + 1, device=self.device, dtype=torch.long)
        win_offsets = torch.arange(W, device=self.device, dtype=torch.long)

        for b0 in range(0, L, batch_size):
            tok = token_idx_all[b0 : b0 + batch_size]
            B = int(tok.numel())

            if T <= W:
                batch_ids = input_ids_full.unsqueeze(0).expand(B, T).clone()
                batch_ids[torch.arange(B, device=self.device), tok] = self.mask_token_id
                with torch.inference_mode():
                    logits = self.model(input_ids=batch_ids).logits
                    logits_at_mask = logits[torch.arange(B, device=self.device), tok, :]
                    lprobs = torch.log_softmax(logits_at_mask.float(), dim=-1)
                final_lprobs[b0 : b0 + B] = lprobs
            else:
                starts = torch.clamp(tok - half, min=0, max=T - W)
                idx = starts.unsqueeze(1) + win_offsets.unsqueeze(0)
                batch_ids = input_ids_full[idx].clone()
                mask_pos = tok - starts
                batch_ids[torch.arange(B, device=self.device), mask_pos] = self.mask_token_id
                with torch.inference_mode():
                    logits = self.model(input_ids=batch_ids).logits
                    logits_at_mask = logits[torch.arange(B, device=self.device), mask_pos, :]
                    lprobs = torch.log_softmax(logits_at_mask.float(), dim=-1)
                final_lprobs[b0 : b0 + B] = lprobs
        return final_lprobs


def score_mutation_delta_logprob(
    global_mut_str: str,
    log_probs_cpu: torch.Tensor,
    tokenizer,
    offset_idx: int,
    wt_structure_seq: str,
):
    """
    Sum of delta log-probabilities for a (possibly multi-site) mutation string.

    For SaProt, we need to handle structure-aware sequences where each amino acid
    is paired with a structure token (e.g., "Ap" = A with structure p).

    Args:
        global_mut_str: Mutation string like "A10G:C20T" (positions are in amino acid space)
        log_probs_cpu: Log probabilities from model [L, V] (L is in structure-aware space)
        tokenizer: SaProt tokenizer
        offset_idx: Position offset (usually 1)
        wt_structure_seq: The structure-aware wildtype sequence (e.g., "ApCyDn...")
    """
    if not global_mut_str or str(global_mut_str).strip() in {"WT", ""}:
        return 0.0

    lprobs = log_probs_cpu.float()
    total = 0.0

    # Check if this is a structure-aware sequence
    is_structure_aware = any(c.islower() for c in wt_structure_seq)

    # Parse structure tokens in fixed pairs to preserve alignment even with "#" padding.
    structure_tokens = []
    if is_structure_aware:
        for i in range(0, len(wt_structure_seq), 2):
            struct = wt_structure_seq[i + 1] if i + 1 < len(wt_structure_seq) else ""
            structure_tokens.append(struct)

    for mutation in str(global_mut_str).split(":"):
        if not mutation or len(mutation) < 3:
            continue

        wt = mutation[0]
        mt = mutation[-1]
        try:
            pos = int(mutation[1:-1])  # This is in amino acid space (1-indexed)
        except ValueError:
            continue

        # Convert amino acid position to 0-indexed
        aa_idx0 = pos - offset_idx

        # For structure-aware sequences, convert amino acid index to structure-aware sequence index
        # Each amino acid occupies 2 characters in the structure-aware sequence
        if is_structure_aware:
            idx0 = aa_idx0  # Position in structure-aware sequence (log_probs indexing)
        else:
            idx0 = aa_idx0

        if idx0 < 0 or idx0 >= lprobs.size(0):
            # Silently skip out-of-bounds positions (warning suppressed for cleaner output)
            continue

        # Optional WT consistency check against structure-aware sequence (aa positions are paired).
        if is_structure_aware:
            wt_pos = aa_idx0 * 2
            if 0 <= wt_pos < len(wt_structure_seq):
                aa_at_pos = wt_structure_seq[wt_pos]
                if aa_at_pos != "#" and aa_at_pos.upper() != wt.upper():
                    print(
                        f"Warning: WT mismatch at pos {pos}: expected {wt}, saw {aa_at_pos}"
                    )

        # Get the structure token for this amino acid position.
        struct_token = ""
        if is_structure_aware and aa_idx0 < len(structure_tokens):
            struct_token = structure_tokens[aa_idx0]

        # For SaProt, tokens are combinations like "Ap" (amino acid + structure)
        wt_token = wt + struct_token if struct_token else wt
        mt_token = mt + struct_token if struct_token else mt

        # Try to get token IDs - if combined token doesn't exist, try just the amino acid
        try:
            wt_id = int(tokenizer.convert_tokens_to_ids(wt_token))
            mt_id = int(tokenizer.convert_tokens_to_ids(mt_token))

            # Check if we got unknown token ID (usually 3)
            unk_id = tokenizer.unk_token_id if hasattr(tokenizer, "unk_token_id") else 3
            if wt_id == unk_id or mt_id == unk_id:
                # Fall back to just amino acid if combined token not found
                wt_id = int(tokenizer.convert_tokens_to_ids(wt))
                mt_id = int(tokenizer.convert_tokens_to_ids(mt))

            delta = (lprobs[idx0, mt_id] - lprobs[idx0, wt_id]).item()
            total += delta

        except Exception as e:
            print(f"Warning: Could not score mutation {mutation}: {e}")
            continue

    return float(total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Local path to a HF SaProt model")
    parser.add_argument(
        "--input-csv",
        default="data/zero_shot/BindingGYM/Binding_substitutions_DMS/5A12_Ang2_fitness_4ZFG.csv",
        help="Input CSV/TSV file",
    )
    parser.add_argument(
        "--output-dir",
        default="results/zero_shot/model_outputs/saprot_bindinggym",
        help="Output directory for result CSVs",
    )
    parser.add_argument("--mode", default="wt", choices=["wt", "masked"], help="Scoring mode")
    parser.add_argument(
        "--focus",
        type=int,
        default=1,
        help="1=drop silent chains, 0=keep all chains",
    )
    parser.add_argument("--offset", type=int, default=1, help="Mutation position offset, usually 1")
    parser.add_argument("--cache-dir", default="results/zero_shot/logits_cache/saprot", help="Cache directory")
    parser.add_argument("--device", default=None, help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--fp16-infer", action="store_true", help="Enable FP16 inference on GPU")
    parser.add_argument(
        "--structure-folder",
        default="data/zero_shot/BindingGYM/structures",
        help="Location of structures",
    )
    parser.add_argument(
        "--foldseek-root",
        default=DEFAULT_FOLDSEEK_ROOT,
        help="Path prefix containing the foldseek binary",
    )
    args = parser.parse_args()

    sep = "\t" if args.input_csv.endswith(".tsv") else ","
    df = pd.read_csv(args.input_csv, sep=sep)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = build_output_path(
        args.output_dir,
        args.input_csv,
        args.model_path,
        mode=args.mode,
        fp16=args.fp16_infer,
        focus=(args.focus == 1),
    )

    if len(df) == 0:
        raise ValueError("No rows found in the dataframe")
    print(f"df shape: {df.shape}", flush=True)

    print("Starting model scoring")
    cache_dir = build_cache_dir(
        args.cache_dir,
        args.model_path,
        fp16=args.fp16_infer,
        focus=(args.focus == 1),
    )

    # Correct parameter order - only model_path and cache_dir as positional args
    engine = SaProtEngine(
        args.model_path,
        cache_dir,
        device=args.device,
        use_fp16_infer=args.fp16_infer,
    )
    tokenizer = engine.tokenizer
    if args.mode != "wt" and tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer does not define a mask token required for masked scoring.")

    all_scores = []
    for poi, poi_df in df.groupby("POI"):
        # load structure for this POI
        pdb_file = poi_df["pdb_file"].iloc[0]
        pdb_path = os.path.join(args.structure_folder, pdb_file)

        # Use ast.literal_eval instead of eval for safety
        wt_seq_dic = poi_df["wildtype_sequence"].iloc[0]
        if isinstance(wt_seq_dic, str):
            wt_seq_dic = ast.literal_eval(wt_seq_dic)

        chains = list(wt_seq_dic.keys())

        # Get structure-informed sequences for the chains
        print(f"Loading structure sequences for POI: {poi}, PDB: {pdb_file}")
        seq_dic = get_struc_seq(
            pdb_path,
            wt_seq_dic=wt_seq_dic,
            python=args.foldseek_root,
            chains=chains,
            process_id=poi,
        )

        # Use combined_seq (index 2) which includes structure information
        # seq_dic[chain] = (seq, struc_seq, combined_seq)
        struct_df = poi_df.copy()
        struct_df["wildtype_sequence"] = [
            {ch: seq_dic[ch][2] for ch in chains if ch in seq_dic}
        ] * len(struct_df)

        # Process each group with the structure-aware sequences
        for group in preprocess_dataframe(
            struct_df,
            wt_col="wildtype_sequence",
            mutant_col="mutant",
            chain_id_col="chain_id",
            poi_col=None,
            focus=(args.focus == 1),
        ):
            combined = group.wt_concat
            print(
                f"Processing sequence of length {len(combined)}, first 50 chars: {combined[:50]}..."
            )

            # Debug: check tokenization
            test_inputs = tokenizer(combined, return_tensors="pt", add_special_tokens=True)
            print(
                f"Tokenized length: {test_inputs['input_ids'].shape[1]} tokens (including special tokens)"
            )
            is_structure_aware = any(c.islower() for c in combined)
            if is_structure_aware:
                expected_aa_count = len(combined) // 2
                print(
                    f"Structure-aware sequence detected: {len(combined)} chars → ~{expected_aa_count} amino acids expected"
                )

            log_probs = engine.get_log_probs(combined, args.mode)
            print(f"Got log_probs shape: {log_probs.shape}")

            for row_idx, mut_global in zip(group.row_indices, group.mutant_global):
                # Pass the structure-aware sequence to the scoring function
                score = score_mutation_delta_logprob(
                    mut_global,
                    log_probs,
                    tokenizer,
                    args.offset,
                    combined,  # Pass the structure-aware sequence
                )
                all_scores.append((row_idx, score))

    # Write scores aligned to original df
    score_series = pd.Series({idx: score for idx, score in all_scores})
    df["saprot_score"] = score_series
    df.to_csv(output_path, index=False)
    print(f"Done. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
