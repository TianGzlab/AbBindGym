#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py - Extended antibody affinity prediction training script with protein language model support

Extended from train_03 with additional models:
- ESM-1v - Meta's Evolutionary Scale Modeling v1 (facebook/esm1v_t33_650M_UR90S_1 and 5 variants)
- ESM-2 - Meta's latest ESM series (facebook/esm2_t33_650M_UR50D and multiple variants)
- ESM3 - EvolutionaryScale's latest ESM model (EvolutionaryScale/esm3-sm-open-v1)
- ProGen2 - NVIDIA's generative protein model (hugohrban/progen2-* series already supported in train_03)
"""

# Import required libraries
import os
# Speed up/avoid optional vision deps from Transformers when unused
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
import random
import json
import numpy as np
import pandas as pd
import re
import math
import hashlib
from datetime import datetime
from typing import Optional, Tuple
from scipy.stats import spearmanr, pearsonr, kendalltau
import torch
from torch import nn
from torch.nn import functional as F
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoConfig, AutoModelForCausalLM
)

def _ensure_esm_tokenizer_patch() -> None:
    """
    Patch the ESM tokenizer stack (ESMC, ESM3, etc.) so it stays compatible with recent
    transformers releases. Besides making legacy special-token properties writable, we
    emulate the older `__getattr__` fallback that certain upstream components relied on.
    """
    try:
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
    except Exception:
        return

    if getattr(EsmSequenceTokenizer, "_antibody_patch_applied", False):
        return

    def _ensure_setter(attr: str, private_name: Optional[str] = None) -> None:
        prop = getattr(EsmSequenceTokenizer, attr, None)
        if not isinstance(prop, property) or prop.fset is not None:
            return
        private_attr = private_name or f"_{attr}"

        def _setter(self, value, _private=private_attr):
            object.__setattr__(self, _private, value)

        setattr(EsmSequenceTokenizer, attr, prop.setter(_setter))

    for token_attr in ["cls_token", "pad_token", "bos_token", "eos_token", "mask_token", "unk_token", "sep_token"]:
        _ensure_setter(token_attr)

    add_prop = getattr(EsmSequenceTokenizer, "additional_special_tokens", None)
    if isinstance(add_prop, property) and add_prop.fset is None:
        def _set_additional(self, value):
            object.__setattr__(self, "_additional_special_tokens", value if value is not None else None)
        EsmSequenceTokenizer.additional_special_tokens = add_prop.setter(_set_additional)

    orig_get_token = EsmSequenceTokenizer._get_token
    _fallback_tokens = {
        "cls_token": "<cls>",
        "bos_token": "<cls>",
        "pad_token": "<pad>",
        "mask_token": "<mask>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "sep_token": "|",
    }

    def _patched_get_token(self, token_name: str) -> str:
        private = f"_{token_name}"
        try:
            val = object.__getattribute__(self, private)
            if val is not None:
                return str(val)
        except AttributeError:
            pass
        try:
            return orig_get_token(self, token_name)
        except AttributeError:
            val = _fallback_tokens.get(token_name)
            if val is not None:
                object.__setattr__(self, private, val)
                return str(val)
            raise

    def _get_id(self, *cands):
        if hasattr(self, "get_vocab"):
            vocab = self.get_vocab()
            for token in cands:
                if token in vocab:
                    return vocab[token]
        for attr in ("token_to_id", "token_to_idx", "encoder", "stoi", "vocab"):
            mapping = getattr(self, attr, None)
            if isinstance(mapping, dict):
                for token in cands:
                    if token in mapping:
                        return mapping[token]
        return None

    def __getattr__(self, name):
        mapping = {
            "pad_token_id": ("<pad>",),
            "bos_token_id": ("<cls>", "<bos>", "<s>"),
            "cls_token_id": ("<cls>", "<bos>", "<s>"),
            "eos_token_id": ("<eos>", "</s>"),
            "sep_token_id": ("|",),
            "unk_token_id": ("<unk>",),
            "mask_token_id": ("<mask>",),
        }
        if name in mapping:
            token_id = _get_id(self, *mapping[name])
            if token_id is not None:
                return token_id
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    EsmSequenceTokenizer._get_token = _patched_get_token
    EsmSequenceTokenizer.__getattr__ = __getattr__
    EsmSequenceTokenizer._antibody_patch_applied = True


_ensure_esm_tokenizer_patch()

# DDP imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from inspect import signature
import logging
import argparse
import warnings
from pathlib import Path as _Path

warnings.simplefilter("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
_DEFAULT_RESULTS_ROOT = _PROJECT_ROOT / "results" / "supervised" / "default"


def _warmup_lazy_modules(net: nn.Module, device: torch.device, sample_seq: str = "ACDEFGHIKLMNPQRSTVWY"):
    """
    Run a tiny forward pass to ensure LazyLinear layers get materialised even when the
    main dataset-based warmup fails (e.g., due to tokenizer quirks). Returns True on success.
    """
    try:
        training = net.training
        net.to(device)
        net.eval()
        if getattr(net, "use_hf_api", True):
            tok = getattr(net, "prot_tokenizer", None)
            if tok is None:
                return False
            dummy = tok(
                [sample_seq],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=getattr(net, "max_length_ab", 512),
            )
            input_ids = dummy["input_ids"].unsqueeze(1).to(device)
            attn = dummy.get("attention_mask")
            if attn is None:
                attn = torch.ones_like(dummy["input_ids"])
            attn = attn.unsqueeze(1).to(device)
            with torch.no_grad():
                _ = net(input_ids, attn, input_ids, attn)
        else:
            with torch.no_grad():
                _ = net([sample_seq], None, [sample_seq], None)
        return True
    except Exception as exc:
        logger.debug(f"[LazyLinear init] Synthetic warmup skipped: {exc}")
        return False
    finally:
        if 'training' in locals():
            net.train(training)

# Define parse_gpus before CLI parsing to avoid NameError
def parse_gpus(gpus_str: str):
    """Parse a GPU list string like '0,1,2,3' into a list of ints. Default to 8 GPUs."""
    if not gpus_str:
        return [0, 1, 2, 3, 4, 5, 6, 7]
    return [int(p.strip()) for p in str(gpus_str).split(',') if p.strip().isdigit()]




parser = argparse.ArgumentParser(description="Affinity model 5-fold CV trainer with resume, model selection, and multi-GPU support (DP/DDP)")
parser.add_argument("--data-path", type=str, default=None, help="Override dataset CSV path (defaults to DATA_PATH)")
parser.add_argument("--dataset-prefix", type=str, default=None, help="Optional prefix for CSV output files (e.g., 'PPB' -> PPB_model_summary.csv)")
parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
parser.add_argument("--epochs", type=int, default=200, help="Training epochs per fold")
parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--model-name", type=str, default=None, help="Display name used for saving models")
parser.add_argument("--net-name", type=str, default=None, help="HF model id to instantiate")
parser.add_argument("--gpus", type=str, default="0", help="GPU ids, e.g. '0' or '0,1'. Empty for CPU")
parser.add_argument(
    "--save-dir",
    type=str,
    default=None,
    help="Optional checkpoint root override (defaults to results_root/checkpoints)",
)
parser.add_argument("--reset-best", action="store_true", help="Ignore previous best when resuming (do not carry best_test_loss)")
parser.add_argument("--results-root", type=str, default=str(_DEFAULT_RESULTS_ROOT), help="Base directory to save CSV and plots (csv/ and plots/)")
parser.add_argument("--workers", type=int, default=4, help="DataLoader workers (recommend <= CPU cores)")
parser.add_argument("--amp", action="store_true", help="Enable mixed precision (fp16) inference/training [NOT RECOMMENDED: use --bf16 instead]")
parser.add_argument("--bf16", action="store_true", default=True, help="Enable bfloat16 mixed precision (RECOMMENDED, default: True based on ablation study)")
parser.add_argument("--tf32", action="store_true", help="Enable TF32 on Ampere/Ada for matmul/cudnn")
parser.add_argument("--predict-batch", type=int, default=None, help="Batch size for prediction/inference")
parser.add_argument("--pad-to-max", action="store_true", default=True, help="Pad tokenization to fixed max_length (keep enabled unless custom collate_fn is used)")
parser.add_argument("--batch-size", type=int, default=32, help="Training batch size (default: 32, optimal for BF16)")
parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
parser.add_argument("--eval-interval", type=int, default=1, help="Validate every N epochs")
parser.add_argument("--clip-grad", type=float, default=0.0, help="Clip grad norm if > 0")
parser.add_argument("--weight-decay", type=float, default=0.0, help="Optimizer weight decay")
parser.add_argument("--scheduler", type=str, choices=["none", "plateau"], default="none", help="LR scheduler type (default: none, RECOMMENDED based on ablation study)")
parser.add_argument("--lrs-factor", type=float, default=0.5, help="LR scheduler factor for plateau")
parser.add_argument("--lrs-patience", type=int, default=5, help="LR scheduler patience for plateau")
parser.add_argument("--min-lr", type=float, default=1e-7, help="Minimum LR for scheduler")
parser.add_argument("--ddp", action="store_true", help="Use DistributedDataParallel (launch via torchrun)")
parser.add_argument("--auto-batch", action="store_true", help="Automatically adjust per-device batch size/grad accumulation to keep the global batch constant across GPU counts")
parser.add_argument("--target-global-batch", type=int, default=None, help="Desired effective global batch when --auto-batch is enabled (defaults to batch_size * grad_accum)")
parser.add_argument("--token-cache-dir", type=str, default=None, help="Optional directory to persist tokenized sequences for reuse")
parser.add_argument("--token-cache-readonly", action="store_true", help="Do not write new cache entries when using --token-cache-dir")
parser.add_argument("--compile", action="store_true", help="Enable torch.compile on the affinity head for faster training")
parser.add_argument("--freeze-backbone", action="store_true", help="Freeze protein language model parameters and train only the downstream heads")
parser.add_argument("--jsonl-log", type=str, default=None, help="Optional JSONL file to append per-epoch metrics (rank0 only)")
parser.add_argument("--save-splits", action="store_true", help="Save generated CV splits to results_root/splits")
parser.add_argument("--splits-file", type=str, default=None, help="Load CV splits JSON and reuse across models")
parser.add_argument("--save-preds", action="store_true", help="Save per-fold predictions to results_root/preds/<model>_<net>/foldX.csv")
parser.add_argument("--run-tag", type=str, default=None, help="Optional tag for this run (used in snapshot CSV filename and column)")
parser.add_argument("--target", type=str, choices=["pkd", "ddg"], default="pkd", help="Training target: 'pkd' for molar-affinity regression or 'ddg' for native AB-Bind delta-delta-G labels")
parser.add_argument("--split", type=str, choices=["random", "group", "stratified_group"], default="random", help="Data split strategy. 'group' uses label-agnostic group-aware splits; legacy 'stratified_group' is kept as an alias.")
parser.add_argument("--token-style", type=str, choices=["raw", "space", "auto"], default="auto", help="How to render text for tokenizer (SaProt strongly prefers 'space').")
parser.add_argument("--use-clean-mask", type=int, default=0, help="1 to drop pad/cls/sep/bos/eos from mask before pooling (default 0: use raw attention_mask)")
parser.add_argument("--use-ss", type=int, default=0, help="1 to pass ss_input_ids (zeros) if supported by backbone")
parser.add_argument("--pooling", type=str, choices=["mean", "cls"], default="mean", help="Pooling strategy for token embeddings")
args, unknown_cli = parser.parse_known_args()

# Flag for using DMS_score as affinity label (BindingGYM)
USING_DMS_AFFINITY = False

# Override device and save directory + DDP initialization
ddp_enabled = bool(getattr(args, 'ddp', False))
is_rank0 = True
gpu_ids = parse_gpus(args.gpus)
WORLD_SIZE = max(1, int(os.environ.get("WORLD_SIZE", "1")))
if ddp_enabled:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    # init process group
    dist.init_process_group(backend=("nccl" if torch.cuda.is_available() else "gloo"), init_method="env://")
    try:
        is_rank0 = (dist.get_rank() == 0)
        WORLD_SIZE = dist.get_world_size()
    except Exception:
        is_rank0 = True
        WORLD_SIZE = max(1, WORLD_SIZE)
    logger.info(f"Selected device: {device}")
else:
    if gpu_ids and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_ids[0]}")
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        logger.info(f"Selected device: {device}")
    else:
        device = torch.device("cpu")
        logger.info("Selected device: CPU")
    WORLD_SIZE = 1

EFFECTIVE_WORLD_SIZE = WORLD_SIZE if ddp_enabled else (len(gpu_ids) if gpu_ids else 1)

ORIGINAL_BATCH = int(getattr(args, 'batch_size', 32))
ORIGINAL_ACCUM = int(getattr(args, 'grad_accum', 1))
if getattr(args, 'auto_batch', False):
    base_global = getattr(args, 'target_global_batch', None)
    if not base_global or base_global <= 0:
        base_global = ORIGINAL_BATCH * max(1, ORIGINAL_ACCUM)
    per_device = max(1, base_global // EFFECTIVE_WORLD_SIZE)
    accum = max(1, math.ceil(base_global / (per_device * EFFECTIVE_WORLD_SIZE)))
    args.batch_size = per_device
    args.grad_accum = accum
    if is_rank0:
        logger.info("Auto batch scheduling enabled: target_global_batch=%s, world_size=%s -> per_device_batch=%s, grad_accum=%s",
                    base_global, EFFECTIVE_WORLD_SIZE, per_device, accum)

if getattr(args, 'predict_batch', None) is None:
    try:
        args.predict_batch = max(1, int(getattr(args, 'batch_size', 32)) // 2)
    except Exception:
        args.predict_batch = 16

JSONL_LOG_PATH = args.jsonl_log if (is_rank0 and getattr(args, 'jsonl_log', None)) else None
TOKEN_CACHE_DIR = getattr(args, 'token_cache_dir', None)
TOKEN_CACHE_READONLY = bool(getattr(args, 'token_cache_readonly', False))
TOKEN_CACHE = None
COMPILE_MODEL = bool(getattr(args, 'compile', False))
FREEZE_BACKBONE = bool(getattr(args, 'freeze_backbone', False))
TORCH_COMPILE_AVAILABLE = hasattr(torch, "compile")

# MODEL_FOLDER will be set below after defaults; CLI may override

if device.type == 'cuda' and args.tf32:
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled for CUDA matmul/cuDNN")
    except Exception as _e:
        logger.warning(f"Failed to enable TF32: {_e}")

AMP_ENABLED = (device.type == 'cuda') and (args.amp or args.bf16)
AMP_DTYPE = torch.bfloat16 if args.bf16 else torch.float16
if device.type == 'cuda':
    amp_mode = ('bf16' if (AMP_ENABLED and AMP_DTYPE==torch.bfloat16) else ('fp16' if (AMP_ENABLED and AMP_DTYPE==torch.float16) else 'off'))
    logger.info(f"AMP mode: {amp_mode}")

# ---------- helpers ----------
def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))

def _dataset_fingerprint(df: pd.DataFrame) -> str:
    """Hash the row order and core fields used by external split indices."""
    columns = [col for col in ("HC", "LC", "Antigen", "Antibody", "affinity") if col in df.columns]
    stable = df.loc[:, columns].fillna("<NA>").astype(str).reset_index(drop=True)
    payload = stable.to_csv(index=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def _as_float(val):
    """Convert tensor or numeric value to float, handling edge cases."""
    if isinstance(val, torch.Tensor):
        return float(val.detach().cpu().item())
    try:
        return float(val)
    except Exception:
        return float("nan")

BATCH_SIZE = args.batch_size
SCALER = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda' and AMP_ENABLED and AMP_DTYPE == torch.float16))

    
# Training outputs follow a shared contract:
#   results_root/checkpoints/<model_key>/fold_<fold>/best.pt
_results_root_for_outputs = _Path(getattr(args, 'results_root', str(_DEFAULT_RESULTS_ROOT))).expanduser().resolve()
if getattr(args, 'save_dir', None):
    CHECKPOINT_ROOT = _Path(args.save_dir).expanduser().resolve()
else:
    CHECKPOINT_ROOT = _results_root_for_outputs / 'checkpoints'
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
logger.info(f"Checkpoint root: {CHECKPOINT_ROOT}")

# Define constants (BATCH_SIZE determined by CLI --batch-size, avoid hardcoded override)
MAX_LENGTH_AB = 512  # Maximum length of antibody sequence
MAX_LENGTH_AG = 512  # Maximum length of antigen sequence
RANDOM_STATE = 314
# Set random seed
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call before training
set_seed(RANDOM_STATE)


# Note: Device already set based on --gpus option after CLI parsing

# Define affinity loss function as mean squared error loss
affinity_loss_fn = nn.MSELoss()

# Set environment variable to avoid multi-threading issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TokenCache:
    """Persistent token cache to avoid redundant HF tokenization."""

    def __init__(self, root: Optional[str], readonly: bool = False):
        self.enabled = bool(root)
        self.readonly = bool(readonly)
        self.root = _Path(root) if root else None
        self._mem = {}
        if self.enabled and self.root and not self.readonly:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str):
        if not self.enabled or self.root is None:
            return None
        return self.root / f"{key}.pt"

    def load(self, key: str):
        if key in self._mem:
            cached = self._mem[key]
        else:
            cached = None
            if self.enabled:
                path = self._path_for(key)
                if path and path.exists():
                    try:
                        cached = torch.load(path, map_location="cpu")
                    except Exception as _e:
                        logger.warning(f"Failed to load token cache entry {path}: {_e}")
                        cached = None
            if cached is not None:
                self._mem[key] = cached
        if cached is None:
            return None
        return {k: v.clone() for k, v in cached.items()}

    def store(self, key: str, value: dict):
        if not self.enabled or self.readonly:
            return
        safe_value = {k: v.detach().cpu() for k, v in value.items()}
        path = self._path_for(key)
        if not path:
            return
        try:
            torch.save(safe_value, path)
            self._mem[key] = safe_value
        except Exception as _e:
            logger.warning(f"Failed to save token cache entry {path}: {_e}")


TOKEN_CACHE = TokenCache(TOKEN_CACHE_DIR, readonly=TOKEN_CACHE_READONLY)


def canonical_net_name(name: str) -> str:
    """
    Standardize model names for CSV records

    Convert local absolute paths to standard format, keep HuggingFace model names unchanged.
    Example: <model-root>/MAGE/model_epoch4 -> custom/MAGE
    """
    name = str(name).strip()

    # Check if it's an absolute path
    if _Path(name).is_absolute():
        # Special handling for MAGE model - unified naming as MAGE
        if 'MAGE' in name or 'mage' in name:
            return "custom/MAGE"

        # Other local models: extract last part as model name
        parts = name.split('/')
        model_name = parts[-1] if parts else name
        # If there's a parent directory name, include it too
        if len(parts) >= 2:
            return f"custom/{parts[-2]}_{model_name}"
        return f"custom/{model_name}"

    return name


def output_model_key(name: str) -> str:
    """Canonical filesystem-safe model key for checkpoints, preds, and plots."""
    return _safe_name(canonical_net_name(name))


def log_jsonl(event: dict):
    """Append an event to the JSONL log if enabled (rank0 only)."""
    if not JSONL_LOG_PATH:
        return
    record = dict(event)
    from datetime import timezone
    record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    try:
        with open(JSONL_LOG_PATH, "a", encoding="utf-8") as _fh:
            _fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as _e:
        logger.warning(f"Failed to write JSONL log {JSONL_LOG_PATH}: {_e}")

def instantiate_model(net_name):
    """
    Instantiate a single model (simplified version: load directly from HuggingFace / corresponding library)
    """
    import os
    from pathlib import Path

    prot_tokenizer = None
    prot_model = None
    use_hf_api = True

    def _truthy_env(name: str) -> bool:
        v = os.environ.get(name)
        if v is None:
            return False
        return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _hf_hub_cache_dir() -> Path | None:
        # Preference order aligned with HF stack behavior.
        # - HF_HUB_CACHE: explicit hub cache
        # - HF_HOME: base dir for huggingface caches, hub lives under it
        # - default: ~/.cache/huggingface/hub
        hub = os.environ.get("HF_HUB_CACHE")
        if hub:
            return Path(hub).expanduser()
        home = os.environ.get("HF_HOME")
        if home:
            return Path(home).expanduser() / "hub"
        return Path.home() / ".cache" / "huggingface" / "hub"

    def _has_cached_config(repo_id: str) -> bool:
        hub_dir = _hf_hub_cache_dir()
        if hub_dir is None:
            return False
        model_dir = hub_dir / f"models--{repo_id.replace('/', '--')}" / "snapshots"
        if not model_dir.exists():
            return False
        try:
            for snap in model_dir.iterdir():
                if (snap / "config.json").is_file():
                    return True
        except Exception:
            return False
        return False

    # ProteinGLM: some environments cache `biomap-research/*` but scripts may pass `proteinglm/*`.
    # In offline mode, resolve to an actually cached repo_id to avoid HF hub lookup failures.
    def _resolve_proteinglm_repo_id(requested: str) -> str:
        alias_map = {
            "proteinglm/proteinglm-3b-mlm": [
                "proteinglm/proteinglm-3b-mlm",
                "biomap-research/proteinglm-3b-mlm",
            ],
            "biomap-research/proteinglm-3b-mlm": [
                "biomap-research/proteinglm-3b-mlm",
                "proteinglm/proteinglm-3b-mlm",
            ],
            "proteinglm/proteinglm-3b-clm": [
                "proteinglm/proteinglm-3b-clm",
                "biomap-research/proteinglm-3b-clm",
            ],
            "biomap-research/proteinglm-3b-clm": [
                "biomap-research/proteinglm-3b-clm",
                "proteinglm/proteinglm-3b-clm",
            ],
        }
        if requested not in alias_map:
            return requested
        if _truthy_env("HF_HUB_OFFLINE") or _truthy_env("TRANSFORMERS_OFFLINE"):
            for cand in alias_map[requested]:
                if _has_cached_config(cand):
                    return cand
        return requested

    requested_net_name = net_name
    net_name = _resolve_proteinglm_repo_id(str(net_name))
    if net_name != requested_net_name:
        logger.warning(f"[model] Offline cache remap: {requested_net_name} -> {net_name}")

    def _log_token_info(tok):
        try:
            try:
                pipe_ids = tok.encode('|', add_special_tokens=False)
            except Exception:
                try:
                    pipe_ids = tok.convert_tokens_to_ids(['|'])
                except Exception:
                    pipe_ids = None
            info = {
                "pad": getattr(tok, "pad_token_id", None),
                "cls": getattr(tok, "cls_token_id", None),
                "sep": getattr(tok, "sep_token_id", None),
                "bos": getattr(tok, "bos_token_id", None),
                "eos": getattr(tok, "eos_token_id", None),
                "unk": getattr(tok, "unk_token_id", None),
                "pipe_ids": pipe_ids,
            }
            logger.info({k: (int(v) if isinstance(v, int) else v) for k, v in info.items()})
        except Exception as _e:
            logger.debug(f"[toklog] failed to log token ids: {_e}")

    try:
        # ==== 1. Classic HF models (ProtBert / ESM / ProtGPT2, etc.) ====
        if net_name in [
            'Rostlab/prot_bert', 'Rostlab/prot_bert_bfd',
            # ESM-2 variants
            'facebook/esm2_t33_650M_UR50D', 'facebook/esm2_t36_3B_UR50D',
            'facebook/esm2_t30_150M_UR50D', 'facebook/esm2_t48_15B_UR50D',
            # ESM-1v variants
            'facebook/esm1v_t33_650M_UR90S_1', 'facebook/esm1v_t33_650M_UR90S_2',
            'facebook/esm1v_t33_650M_UR90S_3', 'facebook/esm1v_t33_650M_UR90S_4',
            'facebook/esm1v_t33_650M_UR90S_5',
            # ProteinGLM (two common repo ids seen in local caches)
            'proteinglm/proteinglm-3b-mlm', 'proteinglm/proteinglm-3b-clm',
            'biomap-research/proteinglm-3b-mlm', 'biomap-research/proteinglm-3b-clm',
            'proteinglm/proteinglm-10b-mlm',
            'nferruz/ProtGPT2', 'jedwang/protein-binding-site-predictor',
            'shashwatsaini/RoBERTa-MLM-For-Protein-Clustering'
        ]:
            try:
                prot_tokenizer = AutoTokenizer.from_pretrained(
                    net_name, trust_remote_code=True
                )
                prot_model = AutoModel.from_pretrained(
                    net_name, trust_remote_code=True
                )
            except ModuleNotFoundError as e:
                if e.name == "deepspeed":
                    raise RuntimeError(
                        f"{net_name} requires the `deepspeed` package. Install it in the current environment, "
                        "e.g. `pip install deepspeed>=0.12`."
                    ) from e
                raise

            if net_name in ['nferruz/ProtGPT2']:
                prot_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                prot_model.resize_token_embeddings(len(prot_tokenizer))

        # ==== 2. ProSST ====
        elif net_name in [
            'AI4Protein/ProSST-1024', 'AI4Protein/ProSST-2048', 'AI4Protein/ProSST-4096',
        ]:
            prot_tokenizer = AutoTokenizer.from_pretrained(net_name, trust_remote_code=True)
            prot_model = AutoModelForMaskedLM.from_pretrained(net_name, trust_remote_code=True)

        elif net_name in [
            'westlake-repl/SaProt_650M_PDB', 'westlake-repl/SaProt_650M_AF2',
            'westlake-repl/SaProt_1.3B_AF2', 'westlake-repl/SaProt_35M_AF2',
        ]:
            prot_tokenizer = AutoTokenizer.from_pretrained(net_name, trust_remote_code=True)
            prot_model = AutoModel.from_pretrained(net_name, trust_remote_code=True)

        # ==== 4. ProtT5 ====
        elif net_name in ['Rostlab/prot_t5_xl_uniref50']:
            from transformers import T5Tokenizer, T5EncoderModel  # lazy import
            prot_tokenizer = T5Tokenizer.from_pretrained(net_name, trust_remote_code=True)
            prot_model = T5EncoderModel.from_pretrained(net_name, trust_remote_code=True)

        elif net_name in ['EvolutionaryScale/esmc-300m-2024-12', 'EvolutionaryScale/esmc-600m-2024-12']:
            from esm.models.esmc import ESMC
            from esm.sdk.api import ESMProtein, LogitsConfig

            class EsmcModel(nn.Module):
                def __init__(self, model_key: str, max_length: int = 512):
                    super().__init__()
                    # Fix for cls_token/eos_token attribute error in ESM SDK with newer transformers
                    import warnings
                    from transformers import PreTrainedTokenizerFast

                    original_setattr = PreTrainedTokenizerFast.__setattr__

                    def patched_setattr(self, name, value):
                        if name in ['cls_token', 'eos_token', 'sep_token', 'pad_token', 'mask_token', 'bos_token']:
                            cls = type(self)
                            if hasattr(cls, name) and isinstance(getattr(cls, name, None), property):
                                return
                        original_setattr(self, name, value)

                    PreTrainedTokenizerFast.__setattr__ = patched_setattr

                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")
                            _ensure_esm_tokenizer_patch()
                            self.client = ESMC.from_pretrained(model_key).to(device)
                    finally:
                        PreTrainedTokenizerFast.__setattr__ = original_setattr

                    self.client.eval()
                    for p in self.client.parameters():
                        p.requires_grad = False
                    self.max_length = max_length
                    self._device = device

                @staticmethod
                def _pre(seq: str) -> str:
                    return re.sub(r"[UZOB]", "X", str(seq))

                @staticmethod
                def _pad_trunc(ten: torch.Tensor, L: int) -> torch.Tensor:
                    cur = ten.shape[1]
                    if cur >= L:
                        return ten[:, :L, :]
                    pad = torch.zeros((ten.shape[0], L - cur, ten.shape[2]), dtype=ten.dtype, device=ten.device)
                    return torch.cat([ten, pad], dim=1)

                @torch.no_grad()
                def forward(self, sequences):
                    if isinstance(sequences, (str, bytes)):
                        sequences = [sequences]
                    outs = []
                    for s in sequences:
                        prot = ESMProtein(sequence=self._pre(s))
                        toks = self.client.encode(prot)
                        logits_out = self.client.logits(toks, LogitsConfig(sequence=True, return_embeddings=True))
                        emb = logits_out.embeddings  # [1, L, C]
                        emb = self._pad_trunc(emb, self.max_length)
                        outs.append(emb.to(self._device, non_blocking=True))
                    return torch.cat(outs, dim=0)  # [B, L, C]

            model_key = 'esmc_300m' if '300m' in net_name else 'esmc_600m'
            prot_model = EsmcModel(model_key=model_key, max_length=MAX_LENGTH_AB)
            use_hf_api = False

        # ==== 6. AIDO.Protein-16B ====
        elif net_name in ['genbio-ai/AIDO.Protein-16B']:
            from modelgenerator.tasks import Embed as AIDOEmbed

            class AidoProtein:
                def __init__(self, model_name: str, max_length: int = 512):
                    import logging as _log
                    _log.info("Loading AIDO 16B model...")
                    config = {"model.backbone": model_name}
                    self.model = AIDOEmbed.from_config(config).to(device)

                    self.model.eval()
                    for p in self.model.parameters():
                        p.requires_grad = False
                    self.max_length = max_length

                def _pre(self, s: str) -> str:
                    return re.sub(r"[UZOB]", "X", str(s))

                def _pad_trunc(self, ten: torch.Tensor, L: int) -> torch.Tensor:
                    cur = ten.shape[1]
                    if cur >= L:
                        return ten[:, :L, :]
                    pad = torch.zeros((ten.shape[0], L - cur, ten.shape[2]), dtype=ten.dtype, device=ten.device)
                    return torch.cat([ten, pad], dim=1)

                def forward(self, sequences):
                    seqs = [self._pre(s) for s in sequences]
                    bat = self.model.transform({"sequences": seqs})
                    output = self.model(bat)
                    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
                        emb = output.last_hidden_state
                    elif isinstance(output, torch.Tensor):
                        emb = output
                    else:
                        raise TypeError(
                            "AIDO model output does not expose last_hidden_state or tensor data; "
                            f"received {type(output)}"
                        )
                    emb = self._pad_trunc(emb, self.max_length)
                    return emb

                def __call__(self, sequences):
                    return self.forward(sequences)

            prot_model = AidoProtein(model_name='aido_protein_16b', max_length=MAX_LENGTH_AB)
            use_hf_api = False

        # ==== 7. VenusPLM ====
        elif net_name in ['AI4Protein/VenusPLM-300M']:
            from vplm import TransformerConfig, TransformerForMaskedLM, VPLMTokenizer

            cfg = TransformerConfig.from_pretrained(net_name, attn_impl="sdpa")
            prot_model = TransformerForMaskedLM.from_pretrained(net_name, config=cfg)
            prot_tokenizer = VPLMTokenizer.from_pretrained(net_name)
            use_hf_api = True

        elif (net_name in [
            'hugohrban/progen2-small', 'hugohrban/progen2-medium', 'hugohrban/progen2-large',
            'hugohrban/progen2-base', 'hugohrban/progen2-xlarge', 'hugohrban/progen2-oas',
            'hugohrban/progen2-BFD90', 'hugohrban/progen2-small-mix7', 'hugohrban/progen2-small-mix7-bidi'
        ] or (os.path.exists(str(net_name)) and os.path.isdir(str(net_name)) and
              os.path.exists(os.path.join(str(net_name), 'config.json')))):

            is_local_progen = os.path.exists(str(net_name)) and os.path.isdir(str(net_name))
            if is_local_progen:
                config_path = os.path.join(str(net_name), 'config.json')
                if os.path.exists(config_path):
                    import json
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)

                    if config_data.get('model_type') == 'progen':
                        modeling_file = os.path.join(str(net_name), 'modeling_progen.py')
                        config_file = os.path.join(str(net_name), 'configuration_progen.py')

                        if os.path.exists(modeling_file) and os.path.exists(config_file):
                            import sys
                            import importlib.util

                            model_dir = os.path.abspath(str(net_name))
                            if model_dir not in sys.path:
                                sys.path.insert(0, model_dir)

                            spec_config = importlib.util.spec_from_file_location("configuration_progen", config_file)
                            progen_config_module = importlib.util.module_from_spec(spec_config)
                            sys.modules['configuration_progen'] = progen_config_module
                            spec_config.loader.exec_module(progen_config_module)

                            spec_model = importlib.util.spec_from_file_location("modeling_progen", modeling_file)
                            progen_model_module = importlib.util.module_from_spec(spec_model)
                            sys.modules['modeling_progen'] = progen_model_module
                            spec_model.loader.exec_module(progen_model_module)

                            ProGenConfig = progen_config_module.ProGenConfig
                            ProGenForCausalLM = progen_model_module.ProGenForCausalLM

                            from transformers import AutoConfig
                            AutoConfig.register("progen", ProGenConfig)
                            AutoModelForCausalLM.register(ProGenConfig, ProGenForCausalLM)

                            logger.info(f"Registered custom ProGen model from {net_name}")

            prot_tokenizer = AutoTokenizer.from_pretrained(net_name, trust_remote_code=True)
            prot_model = AutoModelForCausalLM.from_pretrained(net_name, trust_remote_code=True)

            if prot_tokenizer.pad_token is None:
                if prot_tokenizer.eos_token:
                    prot_tokenizer.pad_token = prot_tokenizer.eos_token
                else:
                    prot_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    prot_model.resize_token_embeddings(len(prot_tokenizer))
            use_hf_api = True

        elif net_name in ['ankh-base', 'ankh-large']:
            print(f"Loading Ankh model from local path: {net_name}")
            from transformers import T5EncoderModel

            local_path_map = {
                'ankh-base': os.environ.get('ANKH_BASE_DIR', 'ankh-base'),
                'ankh-large': os.environ.get('ANKH_LARGE_DIR', 'ankh-large')
            }
            local_path = local_path_map[net_name]

            prot_tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
            prot_model = T5EncoderModel.from_pretrained(local_path, trust_remote_code=True)
            prot_model.eval()
            use_hf_api = True

        # ==== 10. Ankh3 large / xl(HuggingFace) ====
        elif net_name in ['ankh3-large', 'ankh3-xl']:
            print(f"Loading Ankh3 model via Hugging Face: {net_name}")

            hf_model_name_map = {
                'ankh3-large': 'ElnaggarLab/ankh3-large',
                'ankh3-xl':   'ElnaggarLab/ankh3-xl'
            }
            hf_model_name = hf_model_name_map[net_name]

            from transformers import T5Tokenizer, T5EncoderModel  # lazy import
            prot_tokenizer = T5Tokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
            prot_model = T5EncoderModel.from_pretrained(hf_model_name, trust_remote_code=True)

            prot_model.eval()
            use_hf_api = True

        # ==== 11. IgBert ====
        elif net_name in ['Exscientia/IgBert']:
            from transformers import BertTokenizer, BertModel
            prot_tokenizer = BertTokenizer.from_pretrained(net_name, do_lower_case=False)
            prot_model = BertModel.from_pretrained(net_name, add_pooling_layer=False)
            prot_model.eval()
            use_hf_api = True

        # ==== 12. ESM3 ====
        elif net_name in ['EvolutionaryScale/esm3-sm-open-v1']:
            from esm.models.esm3 import ESM3
            from esm.sdk.api import ESMProtein, LogitsConfig

            _esm3_local_dir = os.environ.get(
                "ESM3_LOCAL_DIR"
            )
            if _esm3_local_dir:
                from pathlib import Path as _LocalPath
                try:
                    _esm3_local_path = _LocalPath(_esm3_local_dir).expanduser()
                    if _esm3_local_path.exists():
                        from esm.utils.constants import esm3 as _esm3_constants
                        if not getattr(_esm3_constants, "_antibody_offline_patch", False):
                            _orig_snapshot_download = _esm3_constants.snapshot_download

                            def _offline_snapshot_download(*args, **kwargs):
                                repo_id = kwargs.get("repo_id")
                                if repo_id is None and args:
                                    repo_id = args[0]
                                if repo_id == "EvolutionaryScale/esm3-sm-open-v1":
                                    return _esm3_local_path
                                return _orig_snapshot_download(*args, **kwargs)

                            _esm3_constants.snapshot_download = _offline_snapshot_download
                            _esm3_constants._antibody_offline_patch = True
                            print(f"[ESM3] Using cached files from {_esm3_local_path}")
                except Exception as patch_err:
                    print(f"[WARN] Failed to register local ESM3 cache: {patch_err}")

            class Esm3Model(nn.Module):
                def __init__(self, model_name: str = 'esm3-sm-open-v1', max_length: int = 512):
                    super().__init__()
                    print(f"Loading ESM3 model: {model_name}")
                    self.client = ESM3.from_pretrained(model_name).to(device)
                    self.client.eval()
                    for p in self.client.parameters():
                        p.requires_grad = False
                    self.max_length = max_length
                    self._device = device

                @staticmethod
                def _pre(seq: str) -> str:
                    return re.sub(r"[UZOB]", "X", str(seq))

                @staticmethod
                def _pad_trunc(ten: torch.Tensor, L: int) -> torch.Tensor:
                    cur = ten.shape[1]
                    if cur >= L:
                        return ten[:, :L, :]
                    pad = torch.zeros((ten.shape[0], L - cur, ten.shape[2]), dtype=ten.dtype, device=ten.device)
                    return torch.cat([ten, pad], dim=1)

                @torch.no_grad()
                def forward(self, sequences):
                    if isinstance(sequences, (str, bytes)):
                        sequences = [sequences]
                    outs = []
                    for s in sequences:
                        prot = ESMProtein(sequence=self._pre(s))
                        toks = self.client.encode(prot)
                        logits_out = self.client.logits(toks, LogitsConfig(sequence=True, return_embeddings=True))
                        emb = logits_out.embeddings  # [1, L, C]
                        emb = self._pad_trunc(emb, self.max_length)
                        outs.append(emb.to(self._device, non_blocking=True))
                    return torch.cat(outs, dim=0)

            prot_model = Esm3Model(model_name='esm3-sm-open-v1', max_length=MAX_LENGTH_AB)
            use_hf_api = False

        else:
            raise ValueError(f"Unsupported model name: {net_name}")

        if use_hf_api and hasattr(prot_model, 'eval'):
            prot_model.eval()

        try:
            _max_pos = getattr(prot_model, 'config', None)
            _max_pos = getattr(_max_pos, 'max_position_embeddings', None)
            logger.info(f"[model] max_position_embeddings={_max_pos}")
        except Exception:
            pass

        net = Net01(
            prot_model=prot_model,
            prot_tokenizer=prot_tokenizer,
            max_length_ab=MAX_LENGTH_AB,
            max_length_ag=MAX_LENGTH_AG,
            batch_size=BATCH_SIZE,
            use_hf_api=use_hf_api
        )

        requested_style = getattr(args, 'token_style', None)
        if requested_style in ('raw', 'space', 'auto'):
            token_style = requested_style
        else:
            def _infer_token_style(_net_name: str):
                nm = str(_net_name or '').lower()
                is_progen = 'progen2' in nm or 'progen' in nm or 'mage' in nm or 'gpt' in nm
                if os.path.exists(str(_net_name)) and os.path.isdir(str(_net_name)):
                    config_path = os.path.join(str(_net_name), 'config.json')
                    if os.path.exists(config_path):
                        try:
                            import json
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                            model_type = config.get('model_type', '').lower()
                            if 'progen' in model_type or 'gpt' in model_type:
                                is_progen = True
                        except Exception:
                            pass
                return 'raw' if is_progen else 'space'
            try:
                token_style = _infer_token_style(net_name)
            except Exception:
                token_style = 'space'
        if not use_hf_api:
            token_style = 'raw'

        net.token_style = token_style
        net.sep_token = '|'
        net.use_clean_mask = bool(getattr(args, 'use_clean_mask', 0))
        net.use_ss = bool(getattr(args, 'use_ss', 0))
        net.pooling = 'mean'

        if FREEZE_BACKBONE and hasattr(net, 'prot_model'):
            frozen_params = 0
            for param in getattr(net, 'prot_model').parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_params += 1
            if is_rank0:
                logger.info("Backbone frozen (%s parameters disabled)", frozen_params)

        if use_hf_api and getattr(net, 'prot_tokenizer', None) is not None:
            try:
                _log_token_info(getattr(net, 'prot_tokenizer'))
            except Exception:
                pass

        try:
            bb_params = sum(p.numel() for p in getattr(net, 'prot_model').parameters()) if hasattr(net, 'prot_model') else 0
            bb_trainable = sum(p.numel() for p in getattr(net, 'prot_model').parameters() if p.requires_grad) if hasattr(net, 'prot_model') else 0
            head_trainable = sum(p.numel() for p in getattr(net, 'fc_final_affinity').parameters()) if hasattr(net, 'fc_final_affinity') else 0
            logger.info(f"[freeze] backbone trainable={bb_trainable}/{bb_params}, head trainable={head_trainable}")
        except Exception:
            pass

        try:
            logger.info(
                f"[pooling] strategy={getattr(net, 'pooling', 'mean')}, "
                f"token_style={getattr(net, 'token_style', 'space')}, "
                f"sep_token='{getattr(net, 'sep_token', '|')}'"
            )
            if getattr(net, 'pooling', 'mean') == 'cls' and use_hf_api:
                tok = getattr(net, 'prot_tokenizer', None)
                if tok is not None and getattr(tok, 'cls_token_id', None) is None:
                    logger.info("[pooling] CLS not provided by tokenizer; using first token as CLS surrogate")
        except Exception:
            pass

        return net

    except Exception as e:
        logger.error(f"Failed to instantiate model {net_name}: {e}")
        raise


import pandas as pd

class MyDataset(Dataset):
    _pad_to_max_warned = False  # Class variable for warning control

    def __init__(self, data, prot_tokenizer, max_length_ab, max_length_ag, contain_label=True, use_hf_api=True, token_style: str = 'space', sep_token: str = '[SEP]', token_cache: Optional[TokenCache] = None):
        self.data = data
        self.prot_tokenizer = prot_tokenizer
        self.max_length_ab = max_length_ab
        self.max_length_ag = max_length_ag
        self.contain_label = contain_label
        self.use_hf_api = use_hf_api
        self.token_style = token_style if token_style in ('space', 'raw', 'auto') else 'space'
        self.sep_token = str(sep_token) if sep_token else '[SEP]'
        self._tok_cache = {}
        self.token_cache = token_cache

        # Auto-detect fallback for SaProt-like tokenizers
        if self.use_hf_api and self.prot_tokenizer is not None and self.token_style == 'auto':
            try:
                df = self.data.head(min(32, len(self.data))).reset_index(drop=True)
                ab_texts = []
                ag_texts = []
                for _, row in df.iterrows():
                    hc = str(row.get('HC', '') or '')
                    lc = str(row.get('LC', '') or '')
                    ab = f"{hc}|{lc}" if (hc or lc) else str(row.get('Antibody', '') or '')
                    ab_texts.append(self._clean_sequence(ab))
                    ag_texts.append(self._clean_sequence(str(row.get('Antigen', '') or '')))

                ab_tok = self.prot_tokenizer.batch_encode_plus(
                    ab_texts, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_length_ab, return_tensors='pt')
                ag_tok = self.prot_tokenizer.batch_encode_plus(
                    ag_texts, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_length_ag, return_tensors='pt')
                vt_ab = float(ab_tok['attention_mask'].sum(dim=1).float().mean().item())
                vt_ag = float(ag_tok['attention_mask'].sum(dim=1).float().mean().item())
                logger.info(f"[dataset probe] first pass (raw) valid_tokens_ab_mean={vt_ab:.1f}, valid_tokens_ag_mean={vt_ag:.1f}")
                if vt_ab <= 3.0 or vt_ag <= 3.0:
                    self.token_style = 'space'
                    logger.warning("[SaProt] Detected suspiciously low token counts (<=3). Auto-switching to space-separated tokenization for dataset.")
                else:
                    self.token_style = 'raw'
            except Exception as _e:
                logger.warning(f"[dataset probe] auto token-style detection failed: {_e}; defaulting to 'space'")
                self.token_style = 'space'

        # Token length/truncation diagnostics
        try:
            if self.use_hf_api and self.prot_tokenizer is not None:
                import statistics as _stats
                N = min(64, len(self.data))
                df = self.data.head(N).reset_index(drop=True)
                ab_texts = []
                ag_texts = []
                for _, row in df.iterrows():
                    hc = str(row.get('HC', '') or '')
                    lc = str(row.get('LC', '') or '')
                    ab = f"{hc}|{lc}" if (hc or lc) else str(row.get('Antibody', '') or '')
                    if self.token_style == 'raw':
                        ab_texts.append(self._clean_sequence(ab))
                        ag_texts.append(self._clean_sequence(str(row.get('Antigen', '') or '')))
                    else:
                        ab_texts.append(" ".join(list(self._clean_sequence(ab))))
                        ag_texts.append(" ".join(list(self._clean_sequence(str(row.get('Antigen', '') or '')))))

                def _tok_len(_s: str):
                    try:
                        ids = self.prot_tokenizer.encode(_s, add_special_tokens=True)
                        return (len(ids) if hasattr(ids, '__len__') else int(ids))
                    except Exception:
                        try:
                            enc = self.prot_tokenizer(_s, add_special_tokens=True)
                            ids = enc.get('input_ids', enc) if isinstance(enc, dict) else enc
                            return (len(ids) if hasattr(ids, '__len__') else int(ids))
                        except Exception:
                            return None

                ab_lens = [l for l in (_tok_len(s) for s in ab_texts) if l is not None]
                ag_lens = [l for l in (_tok_len(s) for s in ag_texts) if l is not None]
                if ab_lens and ag_lens:
                    ab_tr = sum(1 for l in ab_lens if l > int(self.max_length_ab)) / float(len(ab_lens))
                    ag_tr = sum(1 for l in ag_lens if l > int(self.max_length_ag)) / float(len(ag_lens))
                    logger.info(f"[tok] trunc_ratio_ab={ab_tr:.3f}, trunc_ratio_ag={ag_tr:.3f}, mean_len_ab={_stats.mean(ab_lens):.1f}, mean_len_ag={_stats.mean(ag_lens):.1f}, max_ab={self.max_length_ab}, max_ag={self.max_length_ag}")
        except Exception as _e:
            logger.debug(f"[tok] truncation diagnostics skipped: {_e}")

    @staticmethod
    def _clean_sequence(seq: str) -> str:
        """Replace uncommon amino acids with X."""
        return re.sub(r"[UZOB]", "X", str(seq))

    @staticmethod
    def _optional_chain_text(value) -> Optional[str]:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        text = str(value).strip()
        if not text or text.lower() == 'nan':
            return None
        return text

    def _extract_chain_texts(self, sample) -> Tuple[Optional[str], Optional[str]]:
        hc = self._optional_chain_text(sample.get('HC', None))
        lc = self._optional_chain_text(sample.get('LC', None))
        antibody = self._optional_chain_text(sample.get('Antibody', None))

        if antibody and (hc is None or lc is None):
            if '|' in antibody:
                hc_part, lc_part = antibody.split('|', 1)
                if hc is None:
                    hc = self._optional_chain_text(hc_part)
                if lc is None:
                    lc = self._optional_chain_text(lc_part)
            elif hc is None and lc is None:
                # Preserve backwards compatibility for legacy single-chain antibody fields.
                hc = antibody

        return hc, lc

    def _prepare_chain_sequence(self, chain_text: Optional[str]):
        if chain_text is None:
            return '' if self.token_style == 'raw' else []
        if self.token_style == 'raw':
            return chain_text
        return list(chain_text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        hc_text, lc_text = self._extract_chain_texts(sample)
        hc_present = torch.tensor(hc_text is not None, dtype=torch.bool)
        lc_present = torch.tensor(lc_text is not None, dtype=torch.bool)

        # Keep a fixed HC/LC/Ag interface. Missing chains are represented explicitly
        # downstream rather than by duplicating the available chain.
        hc_input = self.tokenize_sequence(self._prepare_chain_sequence(hc_text), self.max_length_ab // 2)
        lc_input = self.tokenize_sequence(self._prepare_chain_sequence(lc_text), self.max_length_ab // 2)

        ag_input = self.tokenize_sequence(sample['Antigen'], self.max_length_ag)

        if self.contain_label:
            affinity = torch.tensor(sample['affinity'], dtype=torch.float32)
            return {
                'hc_input': hc_input,
                'lc_input': lc_input,
                'hc_present': hc_present,
                'lc_present': lc_present,
                'ag_input': ag_input,
                'affinity': affinity,
                'idx': index
            }
        else:
            return {
                'hc_input': hc_input,
                'lc_input': lc_input,
                'hc_present': hc_present,
                'lc_present': lc_present,
                'ag_input': ag_input,
                'idx': index
            }

    def pad_or_truncate(self, tensor, target_len=200):
        current_len = tensor.shape[0]
        if current_len >= target_len:
            return tensor[:target_len]
        else:
            padding_shape = [target_len - current_len] + list(tensor.shape[1:])
            padding = torch.zeros(padding_shape, dtype=tensor.dtype)
            return torch.cat([tensor, padding], dim=0)

    def tokenize_sequence(self, sequence, max_length):
        if self.use_hf_api:
            # Raw vs space-separated styles
            if self.token_style == 'raw':
                if isinstance(sequence, (list, tuple)):
                    parts = []
                    for t in sequence:
                        t = str(t)
                        if len(t) == 1:
                            parts.append(self._clean_sequence(t))
                        else:
                            parts.append(t)
                    seq_text = "".join(parts)
                else:
                    seq_text = self._clean_sequence(str(sequence))
            else:
                # Space-separated
                if isinstance(sequence, (list, tuple)):
                    tokens = []
                    for t in sequence:
                        t = str(t)
                        if len(t) == 1:
                            tokens.append(self._clean_sequence(t))
                        else:
                            tokens.append(t)
                    seq_text = " ".join(tokens)
                else:
                    seq_text = " ".join(self._clean_sequence(str(sequence)))

            sequences = [seq_text]
            key = (seq_text, max_length, True)
            cache_hash = None
            cached = self._tok_cache.get(key)
            disk_cached = None
            if cached is None and self.token_cache and self.token_cache.enabled:
                cache_hash = hashlib.sha1(f"{seq_text}|{max_length}|{self.token_style}|{self.use_hf_api}".encode("utf-8")).hexdigest()
                disk_cached = self.token_cache.load(cache_hash)
                if disk_cached is not None:
                    self._tok_cache[key] = disk_cached
                    return {
                        'input_ids': disk_cached['input_ids'],
                        'attention_mask': disk_cached['attention_mask'],
                    }
            if cached is None:
                padding = "max_length"
                # Warn once if pad-to-max is disabled
                try:
                    if not getattr(args, 'pad_to_max', True) and not MyDataset._pad_to_max_warned:
                        logger.warning("--pad-to-max=False requested, but fixed padding is required by current collate. Falling back to 'max_length'.")
                        MyDataset._pad_to_max_warned = True
                except Exception:
                    pass
                sequence_tokens = self.prot_tokenizer.batch_encode_plus(
                    sequences,
                    add_special_tokens=True,
                    padding=padding,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                )
                self._tok_cache[key] = sequence_tokens
            else:
                sequence_tokens = cached
            result = {
                'input_ids': sequence_tokens["input_ids"],
                'attention_mask': sequence_tokens["attention_mask"],
            }
            if cache_hash and self.token_cache:
                self.token_cache.store(cache_hash, {
                    'input_ids': sequence_tokens["input_ids"],
                    'attention_mask': sequence_tokens["attention_mask"],
                })
            return result
        else:
            return {
                'input_ids': sequence,
                'attention_mask': torch.tensor([]),
            }
            
def save_checkpoint(model, optimizer, epoch, best_test_loss, path_savemodel):
    """Save model checkpoint."""
    print("=> Saving checkpoint")
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_test_loss': best_test_loss
    }
    torch.save(state, path_savemodel)

def load_checkpoint(model, optimizer, path_savemodel, device=device):
    """Load model checkpoint."""
    print("=> Loading checkpoint")
    checkpoint = torch.load(path_savemodel, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint.get('epoch', 0)
    best_test_loss = checkpoint.get('best_test_loss', float("inf"))
    return (epoch, best_test_loss)

def train(train_data, test_data, model, path_savemodel, affinity_loss_fn=affinity_loss_fn,
          epochs=100, patience=20, lr=1e-3, num_gpu=1, load_best_test_loss=True,
          grad_accum: int = 1, clip_grad: float = 0.0, weight_decay: float = 0.0,
          eval_interval: int = 1, scheduler: str = "plateau", lrs_factor: float = 0.5,
          lrs_patience: int = 5, min_lr: float = 1e-7, log_context: Optional[dict] = None):
    """Train model with early stopping and learning rate scheduling."""
    log_ctx = dict(log_context or {})
    model.train()
    start_epoch = 0
    best_test_loss = float("inf")
    best_epoch = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler == "plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=lrs_factor, patience=lrs_patience, min_lr=min_lr, verbose=False)
    else:
        lr_scheduler = None

    base_model = getattr(model, 'module', model)
    use_hf_api = base_model.use_hf_api

    if os.path.isfile(path_savemodel):
        start_epoch, best_test_loss = load_checkpoint(model, optimizer, path_savemodel)
        best_epoch = start_epoch
        if not load_best_test_loss:
            best_test_loss = float("inf")

    try:
        dataset = MyDataset(
            data=train_data,
            prot_tokenizer=base_model.prot_tokenizer,
            max_length_ab=base_model.max_length_ab,
            max_length_ag=base_model.max_length_ag,
            use_hf_api=use_hf_api,
            token_style=getattr(base_model, 'token_style', 'space'),
            sep_token=getattr(base_model, 'sep_token', '[SEP]'),
            token_cache=TOKEN_CACHE,
        )

        num_workers = int(getattr(args, 'workers', 4))
        dl_kwargs = dict(
            batch_size=BATCH_SIZE,
            num_workers=num_workers,
            pin_memory=(device.type == 'cuda'),
        )
        if num_workers > 0:
            dl_kwargs.update(dict(persistent_workers=True, prefetch_factor=2))

        if ddp_enabled:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(dataset, shuffle=True)
            dataloader = DataLoader(dataset, sampler=sampler, shuffle=False, **dl_kwargs)
        else:
            dataloader = DataLoader(dataset, shuffle=True, **dl_kwargs)

        train_losses = torch.empty((0, 1), device=device, dtype=torch.float32)
        test_losses = torch.empty((0, 1), device=device, dtype=torch.float32)
        patience_counter = 0

        for epoch in range(start_epoch, epochs):
            total_loss_train = 0.0
            total_count_train = 0
            step = 0
            optimizer.zero_grad(set_to_none=True)

            if ddp_enabled and 'sampler' in locals():
                sampler.set_epoch(epoch)

            pred_sum = 0.0
            pred_sumsq = 0.0
            pred_count = 0

            for batch in dataloader:
                hc_input = batch['hc_input']
                lc_input = batch['lc_input']
                hc_present = batch['hc_present']
                lc_present = batch['lc_present']
                ag_input = batch['ag_input']
                affinity = batch['affinity'].to(device)

                if use_hf_api:
                    if device.type == 'cuda' and AMP_ENABLED:
                        with torch.cuda.amp.autocast(enabled=True, dtype=AMP_DTYPE):
                            predicted_affinity = model(
                                hc_input['input_ids'].to(device, non_blocking=True),
                                hc_input['attention_mask'].to(device, non_blocking=True),
                                lc_input['input_ids'].to(device, non_blocking=True),
                                lc_input['attention_mask'].to(device, non_blocking=True),
                                ag_input['input_ids'].to(device, non_blocking=True),
                                ag_input['attention_mask'].to(device, non_blocking=True),
                                hc_present.to(device, non_blocking=True),
                                lc_present.to(device, non_blocking=True),
                            )
                    else:
                        predicted_affinity = model(
                            hc_input['input_ids'].to(device, non_blocking=True),
                            hc_input['attention_mask'].to(device, non_blocking=True),
                            lc_input['input_ids'].to(device, non_blocking=True),
                            lc_input['attention_mask'].to(device, non_blocking=True),
                            ag_input['input_ids'].to(device, non_blocking=True),
                            ag_input['attention_mask'].to(device, non_blocking=True),
                            hc_present.to(device, non_blocking=True),
                            lc_present.to(device, non_blocking=True),
                        )
                else:
                    predicted_affinity = model(
                        hc_input['input_ids'],
                        hc_input['attention_mask'],
                        lc_input['input_ids'],
                        lc_input['attention_mask'],
                        ag_input['input_ids'],
                        ag_input['attention_mask'],
                        hc_present,
                        lc_present,
                    )

                predicted_affinity = predicted_affinity.view(-1)
                affinity = affinity.to(torch.float32)
                predicted_affinity = predicted_affinity.to(torch.float32)
                loss = affinity_loss_fn(predicted_affinity, affinity)

                bs = affinity.size(0)
                total_loss_train += loss.item() * bs
                total_count_train += bs

                _p = predicted_affinity.detach()
                pred_sum += _p.sum().item()
                pred_sumsq += (_p * _p).sum().item()
                pred_count += _p.numel()

                if SCALER is not None:
                    SCALER.scale(loss / max(1, grad_accum)).backward()
                else:
                    (loss / max(1, grad_accum)).backward()

                step += 1
                if (step % max(1, grad_accum) == 0):
                    if clip_grad and clip_grad > 0:
                        if SCALER is not None:
                            SCALER.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(base_model.parameters(), clip_grad)
                    if SCALER is not None:
                        SCALER.step(optimizer)
                        SCALER.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            mean_train_loss = total_loss_train / max(1, total_count_train)

            if pred_count > 0:
                pred_mean = pred_sum / pred_count
                pred_var = max(0.0, (pred_sumsq / pred_count) - (pred_mean ** 2))
                pred_std = math.sqrt(pred_var)
            else:
                pred_mean = float('nan')
                pred_std = float('nan')

            logger.info(f"Epoch {epoch+1}, train loss: {mean_train_loss:.4f}")
            train_losses = torch.cat((train_losses, torch.tensor([[mean_train_loss]], device=device, dtype=torch.float32)), dim=0)

            epoch_payload = {
                **log_ctx,
                "event": "epoch",
                "epoch": epoch + 1,
                "train_loss": _as_float(mean_train_loss),
                "train_samples": total_count_train,
                "grad_accum": max(1, grad_accum),
                "pred_mean": pred_mean,
                "pred_std": pred_std,
            }

            do_eval = ((epoch + 1) % max(1, eval_interval) == 0)
            if do_eval:
                test_loss = test(test_data, model, affinity_loss_fn, num_gpu)[0]
                test_losses = torch.cat((test_losses, test_loss.reshape(1, 1)), dim=0)
                val_loss = _as_float(test_loss)
                epoch_payload["valid_loss"] = val_loss

                if lr_scheduler is not None:
                    lr_scheduler.step(test_loss.item())

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    patience_counter = 0
                    best_epoch = epoch + 1
                    if (not ddp_enabled) or is_rank0:
                        save_checkpoint(model, optimizer, epoch, best_test_loss, path_savemodel)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"early stop at epoch {epoch+1}")
                        break

            torch.cuda.empty_cache()
            epoch_payload["best_epoch_so_far"] = best_epoch
            epoch_payload["best_valid_loss"] = _as_float(best_test_loss)
            if JSONL_LOG_PATH and is_rank0:
                log_jsonl(epoch_payload)

        print(f"=> Training completed! Best loss is {_as_float(best_test_loss):.4f} at epoch {best_epoch}")
        return (train_losses, test_losses)

    except KeyboardInterrupt:
        print(f"Training interrupted at epoch {epoch}. Best loss at epoch {best_epoch}.")
        raise

def test(train_data, model, affinity_loss_fn=affinity_loss_fn, num_gpu=1):
    """Evaluate model on validation data and return mean loss."""
    model.eval()
    base_model = getattr(model, 'module', model)
    use_hf_api = base_model.use_hf_api
    dataset = MyDataset(
        data=train_data,
        prot_tokenizer=base_model.prot_tokenizer,
        max_length_ab=base_model.max_length_ab,
        max_length_ag=base_model.max_length_ag,
        use_hf_api=use_hf_api,
        token_style=getattr(base_model, 'token_style', 'space'),
        sep_token=getattr(base_model, 'sep_token', '[SEP]'),
        token_cache=TOKEN_CACHE,
    )

    if 'ddp_enabled' in globals() and ddp_enabled:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE, num_workers=0, pin_memory=False)
    else:
        num_workers = int(getattr(args, 'workers', 4))
        dl_kwargs = dict(batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=(device.type == 'cuda'))
        if num_workers > 0:
            dl_kwargs.update(dict(persistent_workers=True, prefetch_factor=2))
        dataloader = DataLoader(dataset, **dl_kwargs)

    total_loss = 0.0
    total_count = 0

    with torch.inference_mode():
        for batch in dataloader:
            hc_input = batch['hc_input']
            lc_input = batch['lc_input']
            hc_present = batch['hc_present']
            lc_present = batch['lc_present']
            ag_input = batch['ag_input']
            affinity = batch['affinity'].to(device, non_blocking=True).to(torch.float32)

            hc_ids = hc_input['input_ids']
            hc_mask = hc_input['attention_mask']
            lc_ids = lc_input['input_ids']
            lc_mask = lc_input['attention_mask']
            ag_ids = ag_input['input_ids']
            ag_mask = ag_input['attention_mask']

            if use_hf_api and device.type == 'cuda' and AMP_ENABLED:
                with torch.cuda.amp.autocast(enabled=True, dtype=AMP_DTYPE):
                    predicted_affinity = model(
                        hc_ids.to(device, non_blocking=True),
                        hc_mask.to(device, non_blocking=True),
                        lc_ids.to(device, non_blocking=True),
                        lc_mask.to(device, non_blocking=True),
                        ag_ids.to(device, non_blocking=True),
                        ag_mask.to(device, non_blocking=True),
                        hc_present.to(device, non_blocking=True),
                        lc_present.to(device, non_blocking=True),
                    )
            elif use_hf_api:
                predicted_affinity = model(
                    hc_ids.to(device, non_blocking=True),
                    hc_mask.to(device, non_blocking=True),
                    lc_ids.to(device, non_blocking=True),
                    lc_mask.to(device, non_blocking=True),
                    ag_ids.to(device, non_blocking=True),
                    ag_mask.to(device, non_blocking=True),
                    hc_present.to(device, non_blocking=True),
                    lc_present.to(device, non_blocking=True),
                )
            else:
                predicted_affinity = model(hc_ids, hc_mask, lc_ids, lc_mask, ag_ids, ag_mask, hc_present, lc_present)

            pred = predicted_affinity.view(-1).to(torch.float32)
            loss_val = affinity_loss_fn(pred, affinity).item()
            bs = affinity.size(0)
            total_loss += loss_val * bs
            total_count += bs

    if 'ddp_enabled' in globals() and ddp_enabled and dist.is_available() and dist.is_initialized():
        t = torch.tensor([total_loss, float(total_count)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss, total_count = t[0].item(), int(t[1].item())

    mean_loss = total_loss / max(1, total_count)
    torch.cuda.empty_cache()
    logger.info(f"valid loss: {mean_loss:.4f}")
    return (torch.tensor([mean_loss], device=device),)

def predict(model, data, batch_size=BATCH_SIZE//4, num_gpu=1):
    """Generate predictions for input data, preserving row indices."""
    model.eval()
    base_model = getattr(model, 'module', model)
    use_hf_api = base_model.use_hf_api
    all_predicted_affinity = []
    dataset = MyDataset(
        data=data,
        prot_tokenizer=base_model.prot_tokenizer,
        max_length_ab=base_model.max_length_ab,
        max_length_ag=base_model.max_length_ag,
        use_hf_api=use_hf_api,
        token_style=getattr(base_model, 'token_style', 'space'),
        sep_token=getattr(base_model, 'sep_token', '[SEP]'),
        token_cache=TOKEN_CACHE,
    )

    num_workers = int(getattr(args, 'workers', 4))
    pred_bs = int(getattr(args, 'predict_batch', max(1, BATCH_SIZE//2)))
    dl_kwargs = dict(
        batch_size=pred_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    if num_workers > 0:
        dl_kwargs.update(dict(persistent_workers=True, prefetch_factor=2))
    dataloader = DataLoader(dataset, **dl_kwargs)

    with torch.inference_mode():
        for batch in dataloader:
            hc_input = batch['hc_input']
            lc_input = batch['lc_input']
            hc_present = batch['hc_present']
            lc_present = batch['lc_present']
            ag_input = batch['ag_input']

            hc_ids = hc_input['input_ids']
            hc_mask = hc_input['attention_mask']
            lc_ids = lc_input['input_ids']
            lc_mask = lc_input['attention_mask']
            ag_ids = ag_input['input_ids']
            ag_mask = ag_input['attention_mask']

            if use_hf_api and device.type == 'cuda' and AMP_ENABLED:
                with torch.cuda.amp.autocast(enabled=True, dtype=AMP_DTYPE):
                    hc_ids = hc_ids.to(device, non_blocking=True)
                    hc_mask = hc_mask.to(device, non_blocking=True)
                    lc_ids = lc_ids.to(device, non_blocking=True)
                    lc_mask = lc_mask.to(device, non_blocking=True)
                    ag_ids = ag_ids.to(device, non_blocking=True)
                    ag_mask = ag_mask.to(device, non_blocking=True)
                    predicted_affinity = model(
                        hc_ids,
                        hc_mask,
                        lc_ids,
                        lc_mask,
                        ag_ids,
                        ag_mask,
                        hc_present.to(device, non_blocking=True),
                        lc_present.to(device, non_blocking=True),
                    )
            elif use_hf_api:
                hc_ids = hc_ids.to(device, non_blocking=True)
                hc_mask = hc_mask.to(device, non_blocking=True)
                lc_ids = lc_ids.to(device, non_blocking=True)
                lc_mask = lc_mask.to(device, non_blocking=True)
                ag_ids = ag_ids.to(device, non_blocking=True)
                ag_mask = ag_mask.to(device, non_blocking=True)
                predicted_affinity = model(
                    hc_ids,
                    hc_mask,
                    lc_ids,
                    lc_mask,
                    ag_ids,
                    ag_mask,
                    hc_present.to(device, non_blocking=True),
                    lc_present.to(device, non_blocking=True),
                )
            else:
                predicted_affinity = model(hc_ids, hc_mask, lc_ids, lc_mask, ag_ids, ag_mask, hc_present, lc_present)

            predicted_affinity = predicted_affinity.view(-1)
            all_predicted_affinity.extend(predicted_affinity.cpu().tolist())
            del hc_ids, hc_mask, lc_ids, lc_mask, ag_ids, ag_mask, predicted_affinity

    torch.cuda.empty_cache()
    predicted_series = pd.Series(all_predicted_affinity, index=data.index)
    return predicted_series

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
    ccc = (2 * r * np.sqrt(vx) * np.sqrt(vy)) / (vx + vy + (mx - my) ** 2 + 1e-12)
    return float(ccc)


def model_test(test_data, net, folder_name, model_name, batch_size=BATCH_SIZE//4, num_gpu=1, scaler=None):
    """Load checkpoint and evaluate on test data, computing comprehensive metrics."""
    path_savemodel = f"{folder_name}/{model_name}"
    model = net.to(device)
    base_model = getattr(model, 'module', model)
    cleaned = None
    try:
        state_dict = torch.load(path_savemodel, map_location=device, weights_only=False)['state_dict']
        cleaned = {k.replace('module.', ''): v for k, v in state_dict.items()}
        base_model.load_state_dict(cleaned, strict=True)
    except Exception as _e:
        logger.warning(f"Strict load failed for {path_savemodel}: {_e}")
        if cleaned is not None:
            try:
                base_model.load_state_dict(cleaned, strict=False)
            except Exception:
                pass
    model.eval()

    target_name = str(getattr(args, 'target', 'pkd')).lower()
    y_true_lab = test_data['affinity'].to_numpy().astype(float)
    y_pred_lab = predict(model, test_data, batch_size=batch_size, num_gpu=num_gpu).to_numpy().astype(float)
    if scaler is not None:
        try:
            y_true_lab = scaler.inverse_transform(y_true_lab.reshape(-1,1)).ravel()
            y_pred_lab = scaler.inverse_transform(y_pred_lab.reshape(-1,1)).ravel()
        except Exception:
            pass

    mse = mean_squared_error(y_true_lab, y_pred_lab)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_pred_lab - y_true_lab)))
    r2 = float(r2_score(y_true_lab, y_pred_lab)) if y_true_lab.size > 1 else np.nan

    sp, sp_p = spearmanr(y_true_lab, y_pred_lab)
    try:
        pr, pr_p = pearsonr(y_true_lab, y_pred_lab)
    except Exception:
        pr, pr_p = (np.nan, np.nan)
    ccc = _concordance_ccc(y_true_lab, y_pred_lab)
    try:
        kt, kt_p = kendalltau(y_true_lab, y_pred_lab)
    except Exception:
        kt, kt_p = (np.nan, np.nan)

    metrics = {
        'MSE': float(mse),
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Spearman': float(sp),
        'Spearman_p': float(sp_p) if sp_p is not None else np.nan,
        'Pearson': float(pr),
        'Pearson_p': float(pr_p) if pr_p is not None else np.nan,
        'KendallTau': float(kt),
        'KendallTau_p': float(kt_p) if kt_p is not None else np.nan,
        'CCC': float(ccc) if not np.isnan(ccc) else np.nan,
        'GMFE': np.nan,
        'P2_within': np.nan,
        'P3_within': np.nan,
    }

    if KD_METRICS_ENABLED:
        kd_true = label_to_kd(np.asarray(y_true_lab), target_name)
        kd_pred = label_to_kd(np.asarray(y_pred_lab), target_name)
        eps = 1e-12
        ratio = np.maximum(kd_true, kd_pred) / np.maximum(np.minimum(kd_true, kd_pred), eps)
        gmfe = float(np.exp(np.mean(np.abs(np.log(np.maximum(kd_pred, eps) / np.maximum(kd_true, eps))))))
        p2 = float(np.mean(ratio <= 2.0))
        p3 = float(np.mean(ratio <= 3.0))
        metrics['GMFE'] = gmfe
        metrics['P2_within'] = p2
        metrics['P3_within'] = p3

    print("Test metrics:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics, y_true_lab, y_pred_lab

if args.data_path is None:
    if __name__ == "__main__":
        raise ValueError("--data-path is required. Please specify the dataset CSV file path.")
    dataset = None
    TARGET = 'pkd'
    USING_DMS_AFFINITY = False
    KD_METRICS_ENABLED = False
else:
    data_path_effective = args.data_path
    logger.info(f"Loading dataset from: {data_path_effective}")
    dataset = pd.read_csv(data_path_effective)
    TARGET = str(getattr(args, 'target', 'pkd')).lower()
    KD_METRICS_ENABLED = False
    _affinity_source_kind = 'unknown'

    rename_map = {}
    cols = set(dataset.columns)
    # Antigen sequence column naming
    if 'Antigen' not in cols:
        if 'antigen_sequence' in cols:
            rename_map['antigen_sequence'] = 'Antigen'
        elif 'Ag_seq' in cols:
            rename_map['Ag_seq'] = 'Antigen'
    if 'HC' not in cols and 'Ab_heavy_chain_seq' in cols:
        rename_map['Ab_heavy_chain_seq'] = 'HC'
    if 'LC' not in cols and 'Ab_light_chain_seq' in cols:
        rename_map['Ab_light_chain_seq'] = 'LC'
    if rename_map:
        dataset = dataset.rename(columns=rename_map)

    _dms_numeric = None
    if 'DMS_score' in dataset.columns:
        _dms_numeric = pd.to_numeric(dataset['DMS_score'], errors='coerce')
        dataset['DMS_score'] = _dms_numeric
        _has_valid_dms = _dms_numeric.notna().sum() > 0
    else:
        _has_valid_dms = False

    _path_hint = "bindinggym" in str(data_path_effective).lower()
    _id_hint = False
    if 'dataset_id' in dataset.columns:
        _id_hint = dataset['dataset_id'].astype(str).str.contains('bindinggym', case=False, na=False).any()
    if 'Ab_name' in dataset.columns and not _id_hint:
        _id_hint = dataset['Ab_name'].astype(str).str.contains('bindinggym', case=False, na=False).any()

    _path_lower = str(data_path_effective).lower()
    _ab_name_series = dataset.get('Ab_name', pd.Series(dtype=str)).astype(str)
    _entry_id_series = dataset.get('entry_id', pd.Series(dtype=str)).astype(str)
    _is_abbind = bool(
        ('ab-bind' in _path_lower) or
        ('abbind' in _path_lower) or
        _ab_name_series.str.contains('ABBind|AB-Bind', case=False, na=False).any() or
        _entry_id_series.str.contains('ABBind-', case=False, na=False).any()
    )

    USING_DMS_AFFINITY = bool(_has_valid_dms and (_path_hint or _id_hint))
    if USING_DMS_AFFINITY:
        logger.info("Detected BindingGYM / DMS_score dataset; using DMS_score directly as affinity target.")

    if _is_abbind and TARGET != 'ddg':
        logger.warning("AB-Bind-like dataset detected with ddG labels present. Use --target ddg to evaluate the native AB-Bind task.")

    if (not USING_DMS_AFFINITY) and TARGET == 'pkd' and 'KD(M)' not in dataset.columns:
        candidate_columns = {
            'IC50 [ug/mL]': ('surrogate_ic50', lambda x: x * (1000 / 150000) * 1e-9),  # MW=150kDa
            'Affinity_Kd [nM]': ('molar_direct', lambda x: x * 1e-9),
            'Affinity_Kd[nM]': ('molar_direct', lambda x: x * 1e-9),
            'Affinity_Kd_nM': ('molar_direct', lambda x: x * 1e-9),  # AlphaSeq format (underscore)
            'Kd_nM': ('molar_direct', lambda x: x * 1e-9),
            'Kd (nM)': ('molar_direct', lambda x: x * 1e-9),
        }

        valid_counts = {}
        for col_name, (col_type, _) in candidate_columns.items():
            if col_name in dataset.columns:
                valid_count = pd.to_numeric(dataset[col_name], errors='coerce').notna().sum()
                valid_counts[col_name] = valid_count

        if valid_counts:
            best_col = max(valid_counts, key=valid_counts.get)
            best_count = valid_counts[best_col]

            if best_count > 0:
                col_type, conversion_func = candidate_columns[best_col]
                dataset['KD(M)'] = pd.to_numeric(dataset[best_col], errors='coerce').apply(conversion_func)
                _affinity_source_kind = col_type
                logger.info(f"Converted {best_col} to KD(M) ({best_count} valid values)")
            else:
                logger.warning("No valid affinity data found in any candidate column")
        else:
            logger.warning("No candidate affinity columns found in dataset")

    dataset['KD(M)'] = pd.to_numeric(dataset.get('KD(M)'), errors='coerce')
    if TARGET == 'pkd' and _affinity_source_kind == 'unknown' and 'KD(M)' in dataset.columns:
        _affinity_source_kind = 'molar_direct'

    dataset = dataset.drop(columns=["Affinity"], errors="ignore")
    dataset = dataset.reset_index(drop=True)

    def kd_to_label(kd: float, target: str):
        if target != 'pkd':
            raise ValueError(f"Unsupported KD conversion target '{target}'.")
        return -np.log10(kd)

    def label_to_kd(label: np.ndarray, target: str):
        if target != 'pkd':
            raise ValueError(f"Unsupported KD conversion target '{target}'.")
        return 10 ** (-label)

    if USING_DMS_AFFINITY:
        dataset['affinity'] = dataset['DMS_score']
        KD_METRICS_ENABLED = False
    elif TARGET == 'ddg':
        ddg_col = None
        for cand in ['ddG', 'ddG(kcal/mol)', 'ddg', 'DDG', 'DeltaDeltaG']:
            if cand in dataset.columns:
                ddg_col = cand
                break
        if ddg_col is None:
            raise ValueError("--target ddg was requested, but no ddG column was found in the dataset.")
        dataset['affinity'] = pd.to_numeric(dataset[ddg_col], errors='coerce')
        KD_METRICS_ENABLED = False
    else:
        dataset['affinity'] = dataset['KD(M)'].apply(lambda x: kd_to_label(x, TARGET))
        KD_METRICS_ENABLED = (_affinity_source_kind == 'molar_direct')

    dataset = dataset.dropna(subset=["Antigen", "affinity"], how="any")

    dataset['Antibody'] = dataset['HC'].fillna('') + '|' + dataset['LC'].fillna('')

    _is_skempi = bool(('skempi' in _path_lower) or _ab_name_series.str.contains('SKEMPI', case=False, na=False).any())
    _is_abcov = bool(('abcov' in _path_lower) or _ab_name_series.str.contains('AbCoV', case=False, na=False).any())
    _is_skempi = _is_skempi or _is_abcov
    if not _is_skempi:
        dataset = dataset.drop_duplicates(subset=['HC', 'LC', 'Antigen'], keep='first')
    dataset = dataset.reset_index(drop=True)

    def _group_ids(df):
        if 'bound_AbAg_PDB_ID' in df.columns:
            return df['bound_AbAg_PDB_ID'].astype(str)
        if {'Ab_PDB_ID','Ag_PDB_ID'}.issubset(df.columns):
            return (df['Ab_PDB_ID'].astype(str) + '|' + df['Ag_PDB_ID'].astype(str))
        if 'Ag_name' in df.columns:
            g = df['Ag_name'].astype(str).str.extract(r'AgSKEMPI-([A-Za-z0-9]+)')[0]
            return g.fillna(df['Ag_name'].astype(str))
        return df['Antigen'].astype(str).str.slice(0, 24)

    def _balanced_group_folds(df, n_splits, random_state):
        groups = pd.Series(_group_ids(df), index=df.index).fillna("__missing_group__").astype(str)
        group_to_indices = {}
        for idx, grp in groups.items():
            group_to_indices.setdefault(grp, []).append(int(idx))
        unique_groups = list(group_to_indices.keys())
        if len(unique_groups) < n_splits:
            raise ValueError(
                f"Cannot build {n_splits} group folds from only {len(unique_groups)} unique groups."
            )
        rng = random.Random(int(random_state))
        group_items = [
            (grp, len(indices), rng.random())
            for grp, indices in group_to_indices.items()
        ]
        group_items.sort(key=lambda item: (-item[1], item[2]))
        fold_groups = [[] for _ in range(n_splits)]
        fold_sizes = [0 for _ in range(n_splits)]
        for grp, size, _ in group_items:
            fold_id = min(range(n_splits), key=lambda i: (fold_sizes[i], len(fold_groups[i])))
            fold_groups[fold_id].append(grp)
            fold_sizes[fold_id] += size
        folds_idx = []
        for grp_list in fold_groups:
            fold_indices = []
            for grp in grp_list:
                fold_indices.extend(group_to_indices[grp])
            folds_idx.append(sorted(fold_indices))
        return folds_idx

    def _group_train_valid_split(df, valid_fraction, random_state):
        groups = pd.Series(_group_ids(df), index=df.index).fillna("__missing_group__").astype(str)
        unique_group_count = groups.nunique()
        if unique_group_count < 2:
            raise ValueError("Group-aware validation split requires at least 2 unique groups.")
        n_valid_folds = min(9, unique_group_count)
        if n_valid_folds < 2:
            raise ValueError("Group-aware validation split requires at least 2 fold partitions.")
        fold_indices = _balanced_group_folds(df, n_valid_folds, random_state)
        target_valid_size = max(1, int(round(len(df) * float(valid_fraction))))
        valid_idx = min(
            fold_indices,
            key=lambda indices: (abs(len(indices) - target_valid_size), len(indices))
        )
        valid_idx_set = set(valid_idx)
        train_idx = [int(idx) for idx in df.index if int(idx) not in valid_idx_set]
        if not train_idx or not valid_idx:
            raise ValueError("Group-aware validation split produced an empty train or validation set.")
        return train_idx, valid_idx

    split_strategy = getattr(args, 'split', None)
    if split_strategy is not None:
        use_strat_group = (split_strategy in {'group', 'stratified_group'})
    else:
        use_strat_group = _is_skempi

    if _is_abcov and 'Ag_name' in dataset.columns:
        ag_counts = dataset['Ag_name'].value_counts()
        dominant_ratio = ag_counts.iloc[0] / len(dataset) if len(ag_counts) > 0 else 0
        if dominant_ratio > 0.8:
            print(f"[INFO] AbCoV dominant antigen ratio: {dominant_ratio:.1%}, using random split instead of group-aware split")
            use_strat_group = False

    print(f"Dataset shape after preprocessing: {dataset.shape}")

class Net01(nn.Module):
    def __init__(
        self,
        prot_model,
        prot_tokenizer,
        max_length_ab,
        max_length_ag,
        batch_size,
        dropout=0.5,
        use_hf_api=True,
    ):
        super(Net01, self).__init__()
        self.prot_model = prot_model
        self.prot_tokenizer = prot_tokenizer
        self.max_length_ab = max_length_ab
        self.max_length_ag = max_length_ag
        self.batch_size = batch_size
        self.dropout = dropout
        self.use_hf_api = use_hf_api
        self.embed_dim = self._infer_embed_dim()

        if self.use_hf_api:
            for param in self.prot_model.parameters():
                param.requires_grad = False

        self._delim_ids = set()
        tok = getattr(self, 'prot_tokenizer', None)
        if tok is not None:
            try:
                ids = tok.encode('|', add_special_tokens=False)
                if isinstance(ids, int):
                    self._delim_ids.add(int(ids))
                else:
                    for i in ids:
                        if isinstance(i, int):
                            self._delim_ids.add(int(i))
            except Exception:
                try:
                    ids = tok.convert_tokens_to_ids(['|'])
                    if isinstance(ids, list):
                        for i in ids:
                            if isinstance(i, int):
                                self._delim_ids.add(int(i))
                    elif isinstance(ids, int):
                        self._delim_ids.add(int(ids))
                except Exception:
                    pass

        input_dim = 3 * self.embed_dim  # HC + LC + Ag (separate pooling per chain)
        self.missing_hc_embedding = nn.Parameter(torch.empty(self.embed_dim))
        self.missing_lc_embedding = nn.Parameter(torch.empty(self.embed_dim))
        nn.init.normal_(self.missing_hc_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.missing_lc_embedding, mean=0.0, std=0.02)
        self.fc_final_affinity = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def _apply_missing_chain_embedding(
        self,
        chain_vec: torch.Tensor,
        chain_present: Optional[torch.Tensor],
        missing_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if chain_present is None:
            return chain_vec
        if not torch.is_tensor(chain_present):
            chain_present = torch.as_tensor(chain_present, dtype=torch.bool, device=chain_vec.device)
        chain_present = chain_present.to(device=chain_vec.device, dtype=torch.bool).reshape(-1, 1)
        missing_vec = missing_embedding.to(device=chain_vec.device, dtype=chain_vec.dtype).unsqueeze(0).expand(chain_vec.size(0), -1)
        return torch.where(chain_present, chain_vec, missing_vec)

    def forward(
        self,
        sequence_input_ids_hc,
        sequence_attention_mask_hc,
        sequence_input_ids_lc,
        sequence_attention_mask_lc,
        sequence_input_ids_ag,
        sequence_attention_mask_ag,
        hc_present=None,
        lc_present=None,
    ):
        def _mean_pool(x, mask):
            m = mask.unsqueeze(1).to(dtype=x.dtype, device=x.device)
            return (x * m).sum(dim=2) / m.sum(dim=2).clamp_min(1.0)

        if not self.use_hf_api:
            x_hc = self.seq_forward(sequence_input_ids_hc, sequence_attention_mask_hc)
            x_lc = self.seq_forward(sequence_input_ids_lc, sequence_attention_mask_lc)
            x_ag = self.seq_forward(sequence_input_ids_ag, sequence_attention_mask_ag)
            m_hc = (x_hc.abs().sum(dim=1, keepdim=True) > 0).to(dtype=x_hc.dtype)
            m_lc = (x_lc.abs().sum(dim=1, keepdim=True) > 0).to(dtype=x_lc.dtype)
            m_ag = (x_ag.abs().sum(dim=1, keepdim=True) > 0).to(dtype=x_ag.dtype)
            hc_vec = (x_hc * m_hc).sum(dim=2) / m_hc.sum(dim=2).clamp_min(1.0)
            lc_vec = (x_lc * m_lc).sum(dim=2) / m_lc.sum(dim=2).clamp_min(1.0)
            ag_vec = (x_ag * m_ag).sum(dim=2) / m_ag.sum(dim=2).clamp_min(1.0)
            hc_vec = self._apply_missing_chain_embedding(hc_vec, hc_present, self.missing_hc_embedding)
            lc_vec = self._apply_missing_chain_embedding(lc_vec, lc_present, self.missing_lc_embedding)
            combined = torch.cat((hc_vec, lc_vec, ag_vec), dim=1)
            return self.fc_final_affinity(combined)

        ids_hc = sequence_input_ids_hc.squeeze(1)
        ids_lc = sequence_input_ids_lc.squeeze(1)
        ids_ag = sequence_input_ids_ag.squeeze(1)
        mask_hc = sequence_attention_mask_hc.squeeze(1).to(dtype=torch.bool)
        mask_lc = sequence_attention_mask_lc.squeeze(1).to(dtype=torch.bool)
        mask_ag = sequence_attention_mask_ag.squeeze(1).to(dtype=torch.bool)

        def _clean_mask(ids, mask, tok):
            bad_ids = set()
            for name in ['pad_token_id','cls_token_id','sep_token_id','bos_token_id','eos_token_id']:
                val = getattr(tok, name, None)
                if val is not None:
                    try:
                        bad_ids.add(int(val))
                    except Exception:
                        pass
            if bad_ids:
                bad = torch.zeros_like(ids, dtype=torch.bool)
                for v in bad_ids:
                    bad |= (ids == v)
                mask = mask & (~bad)
            return mask

        tok = getattr(self, 'prot_tokenizer', None)
        if tok is not None and getattr(self, 'use_clean_mask', False):
            mask_hc = _clean_mask(ids_hc, mask_hc, tok)
            mask_lc = _clean_mask(ids_lc, mask_lc, tok)
            mask_ag = _clean_mask(ids_ag, mask_ag, tok)

        x_hc = self.seq_forward(sequence_input_ids_hc, sequence_attention_mask_hc)
        x_lc = self.seq_forward(sequence_input_ids_lc, sequence_attention_mask_lc)
        x_ag = self.seq_forward(sequence_input_ids_ag, sequence_attention_mask_ag)

        pooling_mode = getattr(self, 'pooling', 'mean')
        if pooling_mode == 'cls':
            hc_vec = x_hc[:, :, 0]
            lc_vec = x_lc[:, :, 0]
            ag_vec = x_ag[:, :, 0]
        else:
            hc_vec = _mean_pool(x_hc, mask_hc)
            lc_vec = _mean_pool(x_lc, mask_lc)
            ag_vec = _mean_pool(x_ag, mask_ag)

        hc_vec = self._apply_missing_chain_embedding(hc_vec, hc_present, self.missing_hc_embedding)
        lc_vec = self._apply_missing_chain_embedding(lc_vec, lc_present, self.missing_lc_embedding)
        combined = torch.cat((hc_vec, lc_vec, ag_vec), dim=1)
        return self.fc_final_affinity(combined)

    def seq_forward(
        self,
        sequence_input_ids,
        sequence_attention_mask,
    ):
        if self.use_hf_api:
            sequence_input_ids = sequence_input_ids.squeeze(1)
            sequence_attention_mask = sequence_attention_mask.squeeze(1)

            forward_kwargs = {
                "input_ids": sequence_input_ids,
                "attention_mask": sequence_attention_mask,
                "output_hidden_states": True,
                "return_dict": True,
            }

            try:
                _sig = signature(self.prot_model.forward)
                _expects_ss = ("ss_input_ids" in _sig.parameters)
            except Exception:
                _expects_ss = False
            _cfg = getattr(self.prot_model, "config", None)
            _cfg_ss_vocab = int(getattr(_cfg, "ss_vocab_size", 0)) if _cfg is not None else 0
            if _expects_ss or _cfg_ss_vocab > 0:
                ss_input_ids = torch.zeros_like(sequence_input_ids, dtype=torch.long, device=sequence_input_ids.device)
                forward_kwargs["ss_input_ids"] = ss_input_ids
            try:
                sig = signature(self.prot_model.forward)
                if "return_dict" not in sig.parameters:
                    forward_kwargs.pop("return_dict", None)
            except Exception:
                pass

            with torch.inference_mode():
                if device.type == 'cuda' and AMP_ENABLED:
                    with torch.cuda.amp.autocast(enabled=True, dtype=AMP_DTYPE):
                        sequence_output = self.prot_model(**forward_kwargs)
                else:
                    sequence_output = self.prot_model(**forward_kwargs)

            if hasattr(sequence_output, 'last_hidden_state') and sequence_output.last_hidden_state is not None:
                emb = sequence_output.last_hidden_state.detach()
            elif hasattr(sequence_output, 'hidden_states') and sequence_output.hidden_states is not None:
                emb = sequence_output.hidden_states[-1].detach()
            else:
                if hasattr(sequence_output, 'logits') and sequence_output.logits is not None:
                    emb = sequence_output.logits.detach()
                else:
                    raise RuntimeError("Model output lacks last_hidden_state/hidden_states/logits; cannot extract embeddings")

            emb = emb.to(sequence_input_ids.device, non_blocking=True)
            if emb.dim() != 3:
                raise RuntimeError(f"Expected 3D embedding, got {emb.dim()}D")
            B, L = sequence_input_ids.shape[:2]
            if emb.shape[0] == B and emb.shape[1] == L:
                pass
            elif emb.shape[0] == L and emb.shape[1] == B:
                emb = emb.permute(1, 0, 2)
            sequence_embedding = emb
        else:
            with torch.inference_mode():
                sequence_embedding = self.prot_model(sequence_input_ids).detach()

        sequence_output = sequence_embedding.permute(0, 2, 1)
        return sequence_output

    def _infer_embed_dim(self) -> int:
        """Infer embedding dimension from backbone model configuration."""
        sources = []
        cfg = getattr(self.prot_model, "config", None)
        if cfg is not None:
            sources.append(cfg)
        sources.append(self.prot_model)

        for attr in ("model", "backbone", "encoder"):
            inner = getattr(self.prot_model, attr, None)
            if inner is not None:
                sources.append(inner)
                inner_cfg = getattr(inner, "config", None)
                if inner_cfg is not None:
                    sources.append(inner_cfg)

        for source in sources:
            for attr in ("hidden_size", "d_model", "model_dim", "n_embd", "embed_dim"):
                value = getattr(source, attr, None)
                if isinstance(value, (int, float)) and value:
                    return int(value)

        for module in sources:
            params = getattr(module, "parameters", None)
            if callable(params):
                for param in params():
                    if isinstance(param, torch.Tensor) and param.ndim >= 2:
                        return int(param.shape[-1])

        raise AttributeError("Unable to infer embed_dim from provided backbone.")

model_names = [
    'prot_bert_01',
    'prot_bert_bfd_01',
    'prot_t5_01',
    'esm2_t30_150m_01',
    'esm2_t33_650m_01',
    'esm2_t36_3b_01',
    # ESM-1v variants (5 models with different seeds for variant effect prediction)
    'esm1v_t33_650m_ur90s_1_01',
    'esm1v_t33_650m_ur90s_2_01',
    'esm1v_t33_650m_ur90s_3_01',
    'esm1v_t33_650m_ur90s_4_01',
    'esm1v_t33_650m_ur90s_5_01',
    'esmc_300m_01',
    'esmc_600m_01',
    # ESM3 - EvolutionaryScale's latest generative model
    'esm3_sm_open_v1_01',
    'ai4protein_prosst_1024_01',
    'ai4protein_prosst_2048_01',
    'ai4protein_prosst_4096_01',
    'venusplm_300m_01',
    'aido_protein_16b_01',
    'protgpt2_01',
    'protein_binding_site_predictor_01',
    'roberta_mlm_for_protein_clustering_01',
    'saprot_1_3b_af2_01',
    'saprot_650m_pdb_01',
    'saprot_650m_af2_01',
    'saprot_35m_af2_01',
    'proteinglm_3b_mlm_01',
    'proteinglm_3b_clm_01',
    'proteinglm_3b_clm_official',  # Official HuggingFace path (proteinglm/proteinglm-3b-clm)
    'hugohrban_progen2-small_01',
    'hugohrban_progen2-medium_01',
    'hugohrban_progen2-large_01',
    'hugohrban_progen2-base_01',
    'hugohrban_progen2-xlarge_01',
    'hugohrban_progen2-oas_01',
    'hugohrban_progen2-BFD90_01',
    'hugohrban_progen2-small-mix7_01',
    'hugohrban_progen2-small-mix7-bidi_01',
    'ankh_base_01',
    'ankh_large_01',
    'ankh3_large_01',
    'ankh3_xl_01',
 ]

net_names = [
    'Rostlab/prot_bert',
    'Rostlab/prot_bert_bfd',
    'Rostlab/prot_t5_xl_uniref50',
    'facebook/esm2_t30_150M_UR50D',
    'facebook/esm2_t33_650M_UR50D',
    'facebook/esm2_t36_3B_UR50D',
    'facebook/esm1v_t33_650M_UR90S_1',
    'facebook/esm1v_t33_650M_UR90S_2',
    'facebook/esm1v_t33_650M_UR90S_3',
    'facebook/esm1v_t33_650M_UR90S_4',
    'facebook/esm1v_t33_650M_UR90S_5',
    'EvolutionaryScale/esmc-300m-2024-12',
    'EvolutionaryScale/esmc-600m-2024-12',
    'EvolutionaryScale/esm3-sm-open-v1',
    'AI4Protein/ProSST-1024',
    'AI4Protein/ProSST-2048',
    'AI4Protein/ProSST-4096',
    'AI4Protein/VenusPLM-300M',
    'genbio-ai/AIDO.Protein-16B',
    'nferruz/ProtGPT2',
    'jedwang/protein-binding-site-predictor',
    'shashwatsaini/RoBERTa-MLM-For-Protein-Clustering',
    'westlake-repl/SaProt_1.3B_AF2',
    'westlake-repl/SaProt_650M_PDB',
    'westlake-repl/SaProt_650M_AF2',
    'westlake-repl/SaProt_35M_AF2',
    'biomap-research/proteinglm-3b-mlm',
    'proteinglm/proteinglm-3b-clm',
    'biomap-research/proteinglm-3b-clm',
    'hugohrban/progen2-small',
    'hugohrban/progen2-medium',
    'hugohrban/progen2-large',
    'hugohrban/progen2-base',
    'hugohrban/progen2-xlarge',
    'hugohrban/progen2-oas',
    'hugohrban/progen2-BFD90',
    'hugohrban/progen2-small-mix7',
    'hugohrban/progen2-small-mix7-bidi',
    'ankh-base',
    'ankh-large',
    'ankh3-large',
    'ankh3-xl',
 ]

def cross_validate_fold(dataset, model_name, net_name, n_splits=5, epochs=200, patience=30, lr=1e-3, device_ids=None, rm_model=False, resume_reset_best=False, fold_splits=None):
    """Perform n-fold cross-validation and return test metrics for each fold."""
    kfold = None if fold_splits is not None else KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    test_result = []

    fold_iter = []
    if fold_splits is not None:
        print(f"[INFO] Using provided splits for {n_splits} folds")
        for f in range(n_splits):
            entry = fold_splits[f]
            train_idx = np.array(entry['train_idx'], dtype=int)
            valid_idx = np.array(entry['valid_idx'], dtype=int)
            test_idx = np.array(entry['test_idx'], dtype=int)
            print(f"[INFO] Fold {f}: train={len(train_idx)}, valid={len(valid_idx)}, test={len(test_idx)}")
            fold_iter.append((f, train_idx, valid_idx, test_idx))
    else:
        for f, (train_full_idx, test_idx) in enumerate(kfold.split(dataset)):
            train_full = dataset.iloc[train_full_idx]
            train_part, valid_part = train_test_split(train_full, train_size=0.8, random_state=RANDOM_STATE)
            train_idx = train_part.index.to_numpy()
            valid_idx = valid_part.index.to_numpy()
            fold_iter.append((f, train_idx, valid_idx, np.array(test_idx)))

    for fold, train_idx, valid_idx, test_idx in fold_iter:
        print(f'Fold {fold}')

        train_data = dataset.iloc[train_idx]
        valid_data = dataset.iloc[valid_idx]
        test_data = dataset.iloc[test_idx]
        train_data.reset_index(drop=True, inplace=True)
        valid_data.reset_index(drop=True, inplace=True)
        test_data.reset_index(drop=True, inplace=True)

        fold_scaler = StandardScaler().fit(train_data[['affinity']])

        def _scale_df(df):
            _df = df.copy()
            _df['affinity'] = fold_scaler.transform(_df[['affinity']])
            return _df

        train_scaled = _scale_df(train_data)
        valid_scaled = _scale_df(valid_data)
        test_scaled_for_loss = _scale_df(test_data)

        net = instantiate_model(net_name)
        primary = device
        if ddp_enabled:
            net.to(primary)
        else:
            if device_ids and len(device_ids) > 0:
                primary = torch.device(f"cuda:{device_ids[0]}")
            net.to(primary)

        if not getattr(net, 'use_hf_api', True):
            _warmup_lazy_modules(net, primary)

        base_model = getattr(net, 'module', net)
        use_hf_api = getattr(base_model, 'use_hf_api', True)

        try:
            init_sample = train_scaled.head(1).reset_index(drop=True)
            init_ds = MyDataset(
                data=init_sample,
                prot_tokenizer=getattr(base_model, 'prot_tokenizer', None),
                max_length_ab=getattr(base_model, 'max_length_ab', 512),
                max_length_ag=getattr(base_model, 'max_length_ag', 512),
                use_hf_api=use_hf_api,
                token_style=getattr(base_model, 'token_style', 'space'),
                sep_token=getattr(base_model, 'sep_token', '[SEP]'),
                token_cache=TOKEN_CACHE,
            )

            net.eval()
            with torch.no_grad():
                batch = init_ds[0]
                hc_input = batch['hc_input']
                lc_input = batch['lc_input']
                hc_present = batch['hc_present']
                lc_present = batch['lc_present']
                ag_input = batch['ag_input']

                def _ensure_batch_tensor(t):
                    if not isinstance(t, torch.Tensor):
                        return t
                    if t.dim() == 0:
                        return t.unsqueeze(0)
                    if t.dim() == 1:
                        return t.unsqueeze(0).unsqueeze(0)
                    if t.dim() == 2:
                        return t.unsqueeze(0)
                    return t

                if use_hf_api:
                    hc_ids = _ensure_batch_tensor(hc_input['input_ids']).to(primary)
                    hc_mask = _ensure_batch_tensor(hc_input['attention_mask']).to(primary)
                    lc_ids = _ensure_batch_tensor(lc_input['input_ids']).to(primary)
                    lc_mask = _ensure_batch_tensor(lc_input['attention_mask']).to(primary)
                    ag_ids = _ensure_batch_tensor(ag_input['input_ids']).to(primary)
                    ag_mask = _ensure_batch_tensor(ag_input['attention_mask']).to(primary)
                    hc_present_b = _ensure_batch_tensor(hc_present).to(primary)
                    lc_present_b = _ensure_batch_tensor(lc_present).to(primary)
                    _ = net(hc_ids, hc_mask, lc_ids, lc_mask, ag_ids, ag_mask, hc_present_b, lc_present_b)
                else:
                    hc_seqs = hc_input['input_ids']
                    lc_seqs = lc_input['input_ids']
                    ag_seqs = ag_input['input_ids']

                    if not isinstance(hc_seqs, (list, tuple)):
                        hc_seqs = [hc_seqs]
                    if not isinstance(lc_seqs, (list, tuple)):
                        lc_seqs = [lc_seqs]
                    if not isinstance(ag_seqs, (list, tuple)):
                        ag_seqs = [ag_seqs]

                    hc_mask = hc_input.get('attention_mask')
                    lc_mask = lc_input.get('attention_mask')
                    ag_mask = ag_input.get('attention_mask')
                    if isinstance(hc_mask, torch.Tensor):
                        hc_mask = hc_mask.to(primary)
                    if isinstance(lc_mask, torch.Tensor):
                        lc_mask = lc_mask.to(primary)
                    if isinstance(ag_mask, torch.Tensor):
                        ag_mask = ag_mask.to(primary)

                    _ = net(hc_seqs, hc_mask, lc_seqs, lc_mask, ag_seqs, ag_mask, hc_present, lc_present)
            net.train()
            torch.cuda.empty_cache()
            if is_rank0:
                logger.info("[LazyLinear init] Successfully initialized model parameters via dummy forward pass")
        except Exception as _e:
            import traceback
            if is_rank0:
                logger.warning(f"[LazyLinear init] Dry-run failed (will continue): {_e}")
                logger.debug(f"[LazyLinear init] Traceback:\n{traceback.format_exc()}")

        lazy_remaining = [
            name for name, param in net.named_parameters()
            if isinstance(param, torch.nn.parameter.UninitializedParameter)
        ]
        if lazy_remaining:
            msg = f"Lazy parameters remain uninitialized after warmup forward pass: {', '.join(lazy_remaining)}"
            if is_rank0:
                logger.error(msg)
            raise RuntimeError(msg)

        if ddp_enabled:
            torch.distributed.barrier()
            if is_rank0:
                logger.info("[DDP] All ranks completed LazyLinear initialization check")

        if COMPILE_MODEL:
            if TORCH_COMPILE_AVAILABLE:
                try:
                    net = torch.compile(net)  # type: ignore[attr-defined]
                    if is_rank0:
                        logger.info("torch.compile enabled for model=%s net=%s fold=%s", model_name, net_name, fold)
                except Exception as _e:
                    logger.warning(f"torch.compile failed, continuing without compilation: {_e}")
            else:
                if is_rank0:
                    logger.warning("torch.compile requested but this PyTorch build does not provide torch.compile")
                globals()['COMPILE_MODEL'] = False

        if ddp_enabled:
            torch.distributed.barrier()
            if is_rank0:
                logger.info("[DDP] All ranks synchronized before DDP wrapping")
            net = DDP(net, device_ids=[primary.index] if primary.type=='cuda' else None, output_device=primary.index if primary.type=='cuda' else None, find_unused_parameters=False)
        elif device_ids and len(device_ids) > 1 and getattr(net, 'use_hf_api', True):
            net = nn.DataParallel(net, device_ids=device_ids)
        else:
            net.to(primary)

        checkpoint_key = output_model_key(net_name)
        checkpoint_dir = CHECKPOINT_ROOT / checkpoint_key / f'fold_{fold}'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path_savemodel = str(checkpoint_dir / 'best.pt')

        if ddp_enabled:
            torch.distributed.barrier()
            if is_rank0:
                logger.info("[DDP] All ranks ready for training/testing")

        if os.path.isfile(path_savemodel):
            print(f"Model file {path_savemodel} already exists!")
        else:
            train(
                train_scaled,
                valid_scaled,
                net,
                path_savemodel=path_savemodel,
                epochs=epochs,
                patience=patience,
                lr=lr,
                num_gpu=(len(device_ids) if device_ids else 1),
                load_best_test_loss=(not resume_reset_best),
                grad_accum=int(getattr(args, 'grad_accum', 1)),
                clip_grad=float(getattr(args, 'clip_grad', 0.0)),
                weight_decay=float(getattr(args, 'weight_decay', 0.0)),
                eval_interval=int(getattr(args, 'eval_interval', 1)),
                scheduler=str(getattr(args, 'scheduler', 'plateau')),
                lrs_factor=float(getattr(args, 'lrs_factor', 0.5)),
                lrs_patience=int(getattr(args, 'lrs_patience', 5)),
                min_lr=float(getattr(args, 'min_lr', 1e-7)),
                log_context={
                    "model": str(model_name),
                    "net": str(net_name),
                    "fold": int(fold),
                },
            )

        metrics_and_arrays = model_test(
            test_data=test_scaled_for_loss,
            net=net,
            folder_name=str(checkpoint_dir),
            model_name="best.pt",
            scaler=fold_scaler,
        )

        metrics, y_true_aff, y_pred_aff = metrics_and_arrays
        print(f"Fold {fold} metrics summary: MSE={metrics['MSE']:.4f}, Spearman={metrics['Spearman']:.4f}, RMSE={metrics['RMSE']:.4f}")

        test_result.append(metrics)

        try:
            if getattr(args, 'save_preds', False) and (not ddp_enabled or is_rank0):
                preds_root = _Path(getattr(args, 'results_root', str(_DEFAULT_RESULTS_ROOT))) / 'preds'
                safe_net = output_model_key(net_name)
                out_dir = preds_root / f"{safe_net}"
                out_dir.mkdir(parents=True, exist_ok=True)
                y_pred_unscaled = np.asarray(y_pred_aff, dtype=float)
                if KD_METRICS_ENABLED:
                    kd_pred = label_to_kd(np.asarray(y_pred_unscaled), TARGET)
                    kd_true = test_data.get('KD(M)', pd.Series(index=test_data.index, dtype=float))
                else:
                    kd_pred = np.full_like(np.asarray(y_pred_unscaled, dtype=float), np.nan, dtype=float)
                    kd_true = pd.Series(index=test_data.index, dtype=float)
                out_df = pd.DataFrame({
                    'Fold': fold + 1,
                    'idx': test_data.index,
                    'Antibody': test_data.get('Antibody', pd.Series(index=test_data.index, dtype=object)),
                    'Antigen': test_data.get('Antigen', pd.Series(index=test_data.index, dtype=object)),
                    'KD(M)_true': kd_true,
                    'KD(M)_pred': kd_pred,
                    'affinity_true_label': y_true_aff,
                    'affinity_pred_label': y_pred_aff,
                })
                out_csv = out_dir / f"fold{fold}.csv"
                out_df.to_csv(out_csv, index=False)
                print(f"Saved predictions: {out_csv}")
        except Exception as _e:
            print(f"[WARN] Failed to save predictions for fold {fold}: {_e}")

        try:
            import matplotlib.pyplot as _plt
            _plots_dir = _Path(getattr(args, 'results_root', str(_DEFAULT_RESULTS_ROOT))) / 'plots'
            _plots_dir.mkdir(parents=True, exist_ok=True)

            fig, ax = _plt.subplots(1, 1, figsize=(5, 5))
            ax.scatter(y_true_aff, y_pred_aff, s=12, alpha=0.6)
            mn = float(min(np.min(y_true_aff), np.min(y_pred_aff)))
            mx = float(max(np.max(y_true_aff), np.max(y_pred_aff)))
            ax.plot([mn, mx], [mn, mx], 'r--', lw=1)
            unit_map = {'pkd': 'pKd', 'ddg': 'ddG (kcal/mol)'}
            if USING_DMS_AFFINITY:
                xylab = 'DMS score'
            else:
                xylab = unit_map.get(TARGET, 'label')
            ax.set_xlabel(f'True {xylab}')
            ax.set_ylabel(f'Pred {xylab}')
            ax.set_title(f'{model_name}|{net_name} Fold {fold} (R2={metrics["R2"]:.2f}, RMSE={metrics["RMSE"]:.2f})')
            fig.tight_layout()
            safe_net = output_model_key(net_name)
            out_png = _plots_dir / f'{safe_net}_fold{fold}_scatter.png'
            fig.savefig(out_png, dpi=150)
            _plt.close(fig)

            resid = y_pred_aff - y_true_aff
            fig, axes = _plt.subplots(1, 2, figsize=(10, 4))
            axes[0].hist(resid, bins=30, color='#4C78A8', alpha=0.8)
            axes[0].set_title('Residual histogram')
            axes[0].set_xlabel(f'Pred - True ({xylab})')
            axes[0].grid(axis='y', linestyle=':', alpha=0.4)

            axes[1].scatter(y_pred_aff, resid, s=10, alpha=0.6, color='#F58518')
            axes[1].axhline(0.0, color='r', linestyle='--', lw=1)
            axes[1].set_title('Residual vs Pred')
            axes[1].set_xlabel(f'Pred affinity ({xylab})')
            axes[1].set_ylabel('Residual')
            axes[1].grid(axis='y', linestyle=':', alpha=0.4)
            fig.tight_layout()
            out_png = _plots_dir / f'{safe_net}_fold{fold}_residuals.png'
            fig.savefig(out_png, dpi=150)
            _plt.close(fig)
        except Exception as _e:
            print(f"[WARN] Failed to save diagnostic plots for fold {fold}: {_e}")

        if rm_model:
            try:
                os.remove(path_savemodel)
                print(f"Model file {path_savemodel} deleted")
            except OSError as e:
                print(f"Failed to delete model file {path_savemodel} error: {e}")
        del train_data, valid_data, test_data, net
        torch.cuda.empty_cache()

    if test_result:
        keys = list(test_result[0].keys())
        avg = {k: float(np.nanmean([d.get(k, np.nan) for d in test_result])) for k in keys}
        print(f"{n_splits}-Fold metrics average:\n" + json.dumps(avg, ensure_ascii=False, indent=2))

    return test_result

def calculate_cross_validate(dataset, model_names, net_names, n_splits=5, epochs=200, patience=30, lr=1e-3, device_ids=None, rm_model=False, resume_reset_best=False, fold_splits=None):
    """Run cross-validation for multiple model/net pairs and collect results."""
    results = []
    for model_name, net_name in zip(model_names, net_names):
        try:
            this_result = cross_validate_fold(
                dataset, model_name, net_name,
                n_splits=n_splits, epochs=epochs, patience=patience, lr=lr,
                device_ids=device_ids, rm_model=rm_model, resume_reset_best=resume_reset_best, fold_splits=fold_splits
            )
        except NotImplementedError as e:
            logger.warning(f"Skip {net_name}: {e}")
            this_result = []
        except Exception as e:
            logger.exception(f"Failed {net_name}: {e}")
            this_result = []
        results.append({'model_name': model_name, 'net_name': net_name, 'results': this_result})
    return results

if __name__ == "__main__":
    dataset_cross = dataset.reset_index(drop=True)
    dataset_fingerprint = _dataset_fingerprint(dataset_cross)

    folds = int(getattr(args, 'folds', 5))
    epochs = int(getattr(args, 'epochs', 200))
    patience = int(getattr(args, 'patience', 30))
    lr = float(getattr(args, 'lr', 1e-4))

    if args.model_name and args.net_name:
        model_names = [args.model_name]
        net_names = [args.net_name]

    _results_root = _Path(getattr(args, 'results_root', str(_DEFAULT_RESULTS_ROOT)))
    (_results_root / 'splits').mkdir(parents=True, exist_ok=True)
    splits_to_use = None
    if getattr(args, 'splits_file', None):
        try:
            with _Path(args.splits_file).open('r', encoding='utf-8') as f:
                loaded = json.load(f)
            splits_size = loaded.get('meta', {}).get('size')
            actual_size = len(dataset_cross)
            if splits_size is None:
                raise ValueError(f"splits_file missing meta.size: {args.splits_file}")
            if int(splits_size) != int(actual_size):
                raise ValueError(
                    f"splits meta.size={splits_size} but loaded dataset has {actual_size} rows. "
                    f"Use the exact same CSV (and row order) that was used to generate {args.splits_file}."
                )
            splits_fingerprint = loaded.get('meta', {}).get('fingerprint')
            if splits_fingerprint is None:
                logger.warning(
                    "splits_file has no meta.fingerprint; only row count can be checked. "
                    "Regenerate splits with --save-splits to guard against row-order drift."
                )
            elif str(splits_fingerprint) != dataset_fingerprint:
                raise ValueError(
                    f"splits meta.fingerprint does not match the current preprocessed dataset. "
                    f"Use the exact same CSV (and row order) that was used to generate {args.splits_file}."
                )
            splits_to_use = loaded.get('folds')
            if not splits_to_use:
                raise ValueError(f"splits_file has no folds: {args.splits_file}")
        except Exception as _e:
            raise RuntimeError(f"Failed to load splits_file {args.splits_file}: {_e}") from _e
    if splits_to_use is None:
        tmp = []
        if folds >= 2:
            if use_strat_group:
                try:
                    outer_folds = _balanced_group_folds(dataset_cross, folds, RANDOM_STATE)
                except ValueError as exc:
                    logger.warning(
                        "Falling back to random KFold because group-aware outer split is unavailable: %s",
                        exc,
                    )
                    outer_folds = None
                if outer_folds is not None:
                    for fold_idx, test_idx in enumerate(outer_folds):
                        test_idx_set = set(test_idx)
                        train_full_idx = [int(idx) for idx in dataset_cross.index if int(idx) not in test_idx_set]
                        train_full = dataset_cross.iloc[train_full_idx]
                        try:
                            train_idx, valid_idx = _group_train_valid_split(
                                train_full,
                                valid_fraction=0.1,
                                random_state=RANDOM_STATE + fold_idx + 1,
                            )
                            train_part = dataset_cross.loc[train_idx]
                            valid_part = dataset_cross.loc[valid_idx]
                        except ValueError as exc:
                            logger.warning(
                                "Falling back to random train/valid split inside group-aware fold because %s",
                                exc,
                            )
                            train_part, valid_part = train_test_split(
                                train_full,
                                train_size=0.9,
                                random_state=RANDOM_STATE + fold_idx + 1,
                            )
                        tmp.append({
                            'train_idx': train_part.index.to_list(),
                            'valid_idx': valid_part.index.to_list(),
                            'test_idx': list(map(int, test_idx)),
                        })
                else:
                    kf = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
                    for train_full_idx, test_idx in kf.split(dataset_cross):
                        train_full = dataset_cross.iloc[train_full_idx]
                        train_part, valid_part = train_test_split(train_full, train_size=0.8, random_state=RANDOM_STATE)
                        tmp.append({
                            'train_idx': train_part.index.to_list(),
                            'valid_idx': valid_part.index.to_list(),
                            'test_idx': list(map(int, test_idx)),
                        })
            else:
                kf = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
                for train_full_idx, test_idx in kf.split(dataset_cross):
                    train_full = dataset_cross.iloc[train_full_idx]
                    train_part, valid_part = train_test_split(train_full, train_size=0.8, random_state=RANDOM_STATE)
                    tmp.append({
                        'train_idx': train_part.index.to_list(),
                        'valid_idx': valid_part.index.to_list(),
                        'test_idx': list(map(int, test_idx)),
                    })
        else:
            # folds == 1: build a single split (80% train_full / 20% test; within train_full 80% train / 20% valid)
            idx_all = np.arange(len(dataset_cross))
            train_full_idx, test_idx = train_test_split(idx_all, train_size=0.8, random_state=RANDOM_STATE, shuffle=True)
            train_idx, valid_idx = train_test_split(train_full_idx, train_size=0.8, random_state=RANDOM_STATE, shuffle=True)
            tmp.append({
                'train_idx': list(map(int, train_idx)),
                'valid_idx': list(map(int, valid_idx)),
                'test_idx': list(map(int, test_idx)),
            })
        splits_to_use = tmp
        if getattr(args, 'save_splits', False) and is_rank0:
            dataset_name = _safe_name(_Path(str(data_path_effective)).stem)
            splits_path = _results_root / 'splits' / f"{dataset_name}_k{folds}_seed{RANDOM_STATE}.json"
            try:
                with splits_path.open('w', encoding='utf-8') as f:
                    json.dump(
                        {
                            'meta': {
                                'n_splits': folds,
                                'seed': RANDOM_STATE,
                                'size': len(dataset_cross),
                                'fingerprint': dataset_fingerprint,
                            },
                            'folds': splits_to_use,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                print(f"[SPLITS] Saved to {splits_path}")
            except Exception as _e:
                print(f"[WARN] Failed to save splits {splits_path}: {_e}")

    results_cross = calculate_cross_validate(
        dataset_cross, model_names, net_names,
        n_splits=folds, epochs=epochs, patience=patience, lr=lr,
        device_ids=(None if ddp_enabled else gpu_ids), rm_model=False, resume_reset_best=bool(getattr(args, 'reset_best', False)), fold_splits=splits_to_use
    )

    for result in results_cross:
        print(f"Model: {result['model_name']}, Net: {result['net_name']}, Results: {result['results']}")

    if is_rank0:
        _results_root = _Path(getattr(args, 'results_root', str(_DEFAULT_RESULTS_ROOT)))
        _csv_dir = _results_root / 'csv'
        _plots_dir = _results_root / 'plots'
        _csv_dir.mkdir(parents=True, exist_ok=True)
        _plots_dir.mkdir(parents=True, exist_ok=True)

        try:
            dataset_name = _safe_name(_Path(str(data_path_effective)).stem)
        except Exception:
            dataset_name = 'dataset'

        if getattr(args, 'dataset_prefix', None):
            csv_basename = f'{args.dataset_prefix}_{dataset_name}_model_summary.csv'
        else:
            csv_basename = f'{dataset_name}_model_summary.csv'
        csv_path = _csv_dir / csv_basename

        try:
            from datetime import datetime as _dt
            _run_tag = str(getattr(args, 'run_tag', '') or os.environ.get('RUN_TAG') or os.environ.get('RUN_STAMP') or _dt.now().strftime('%Y%m%d_%H%M%S'))
        except Exception:
            _run_tag = 'run'

        records = []

        for result in results_cross:
            model_name = result['model_name']
            net_name = result['net_name']
            fold_results = result['results'] or []

            for fold_idx, metrics in enumerate(fold_results):
                row = {
                    'Model': model_name,
                    'Net': canonical_net_name(net_name),
                    'Fold': fold_idx + 1,
                    'RunTag': _run_tag,
                }
                for k, v in metrics.items():
                    row[k] = v
                try:
                    row['Pooling'] = str(getattr(args, 'pooling', 'mean'))
                except Exception:
                    pass
                records.append(row)

        df_results = pd.DataFrame(records)
        if csv_path.exists():
            try:
                df_prev = pd.read_csv(csv_path)
            except Exception:
                df_prev = pd.DataFrame()
            df_all = pd.concat([df_prev, df_results], ignore_index=True)
            if 'Fold' in df_all.columns:
                df_all['Fold'] = pd.to_numeric(df_all['Fold'], errors='coerce').fillna(0).astype(int)
            keys = [c for c in ['Model','Net','Fold'] if c in df_all.columns]
            if keys:
                df_all = df_all.drop_duplicates(subset=keys, keep='last')
        else:
            df_all = df_results
        df_all.to_csv(str(csv_path), index=False)
        print(f"Results successfully saved to {csv_path}")

        if not df_all.empty and 'Spearman' in df_all.columns:
            rank_df = (
                df_all.groupby(['Model', 'Net'])['Spearman']
                .mean()
                .reset_index()
                .rename(columns={'Spearman': 'Spearman_mean'})
                .sort_values('Spearman_mean', ascending=False)
            )
            if getattr(args, 'dataset_prefix', None):
                rank_basename = f'{args.dataset_prefix}_{dataset_name}_ranking_by_spearman.csv'
            else:
                rank_basename = f'{dataset_name}_ranking_by_spearman.csv'
            rank_path = str(_csv_dir / rank_basename)
            rank_df.to_csv(rank_path, index=False)
            print(f"Ranking by Spearman saved to {rank_path}")

        try:
            import matplotlib.pyplot as _plt
            import seaborn as _sns
            if not df_all.empty:
                for (model_name, net_name), df_grp in df_all.groupby(['Model', 'Net']):
                    fig, axes = _plt.subplots(1, 3, figsize=(16, 4))
                    fig.suptitle(f"{model_name} | {net_name}")
                    _sns.barplot(x='Fold', y='MSE', data=df_grp, ax=axes[0], color="#4C78A8")
                    axes[0].set_title('MSE by Fold')
                    axes[0].grid(axis='y', linestyle=':', alpha=0.4)
                    _sns.barplot(x='Fold', y='Spearman', data=df_grp, ax=axes[1], color="#54A24B")
                    axes[1].set_title('Spearman by Fold')
                    axes[1].set_ylim(-1.0, 1.0)
                    axes[1].grid(axis='y', linestyle=':', alpha=0.4)
                    if 'MAE' in df_grp.columns:
                        _sns.barplot(x='Fold', y='MAE', data=df_grp, ax=axes[2], color="#F58518")
                        axes[2].set_title('MAE by Fold')
                        axes[2].grid(axis='y', linestyle=':', alpha=0.4)
                    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                    safe_net = output_model_key(net_name)
                    out_png = _plots_dir / f"{safe_net}_metrics.png"
                    fig.savefig(out_png, dpi=150)
                    _plt.close(fig)
                    print(f"Saved plot: {out_png}")
        except Exception as _e:
            print(f"[WARN] Failed to save plots: {_e}")

    try:
        if 'ddp_enabled' in globals() and ddp_enabled and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
