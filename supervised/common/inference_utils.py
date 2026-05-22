#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared inference utilities extracted from `train.py`.

These helpers keep inference and evaluation behavior aligned with the training
stack.

Main components:
- `load_model_components`
- `Net01`
- `MyDataset`
"""

# ============================================================================
# Imports mirrored from `train.py`.
# ============================================================================

import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import random
import json
import numpy as np
import pandas as pd
import re
import math
import hashlib
from datetime import datetime
from inspect import signature
from typing import Optional, Tuple
from scipy.stats import spearmanr, pearsonr, kendalltau

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoConfig,
    AutoModelForCausalLM
)

# ESM tokenizer patch
def _ensure_esm_tokenizer_patch() -> None:
    """Patch ESM tokenizer for compatibility"""
    try:
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
    except Exception:
        return

    if getattr(EsmSequenceTokenizer, "_antibody_patch_applied", False):
        return

    # Add __getattr__ method to EsmSequenceTokenizer
    def _getattr(self, name):
        """Fallback for attribute access"""
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    EsmSequenceTokenizer.__getattr__ = _getattr
    EsmSequenceTokenizer._antibody_patch_applied = True

_ensure_esm_tokenizer_patch()

# ============================================================================
# Logging setup
# ============================================================================

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Device configuration
# ============================================================================

# Simplified device setup used by inference-only scripts.
import torch
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    logger.info(f"Selected device: cuda:0")
else:
    device = torch.device("cpu")
    logger.info("Selected device: CPU")

# AMP (Automatic Mixed Precision) defaults for inference.
AMP_ENABLED = (device.type == 'cuda')
AMP_DTYPE = torch.bfloat16
logger.info(f"AMP mode: {'bf16' if AMP_ENABLED else 'disabled'}")

# ============================================================================
# Constants mirrored from `train.py`.
# ============================================================================

MAX_LENGTH_AB = 512
MAX_LENGTH_AG = 512
BATCH_SIZE = 64

# ============================================================================
# `load_model_components` extracted from the model-instantiation path.
# ============================================================================

def load_model_components(net_name):
    """Instantiate one model from HuggingFace or the corresponding library."""
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
        # ==== 1. Standard HF models (ProtBert / ESM / ProtGPT2, etc.) ====
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

        # ==== 3. SaProt family ====
        elif net_name in [
            'westlake-repl/SaProt_650M_PDB', 'westlake-repl/SaProt_650M_AF2',
            'westlake-repl/SaProt_1.3B_AF2', 'westlake-repl/SaProt_35M_AF2',
        ]:
            prot_tokenizer = AutoTokenizer.from_pretrained(net_name, trust_remote_code=True)
            prot_model = AutoModel.from_pretrained(net_name, trust_remote_code=True)

            # Audit special token ids and '|' token
            try:
                pipe_enc = prot_tokenizer("|", add_special_tokens=False)
                pipe_ids = pipe_enc.get('input_ids', pipe_enc) if isinstance(pipe_enc, dict) else pipe_enc
            except Exception:
                pipe_ids = None
            logger.info({
                "pad": getattr(prot_tokenizer, "pad_token_id", None),
                "cls": getattr(prot_tokenizer, "cls_token_id", None),
                "sep": getattr(prot_tokenizer, "sep_token_id", None),
                "bos": getattr(prot_tokenizer, "bos_token_id", None),
                "eos": getattr(prot_tokenizer, "eos_token_id", None),
                "unk": getattr(prot_tokenizer, "unk_token_id", None),
                "pipe_ids": pipe_ids,
            })
            try:
                _log_token_info(prot_tokenizer)
            except Exception:
                pass

        # ==== 4. ProtT5 ====
        elif net_name in ['Rostlab/prot_t5_xl_uniref50']:
            from transformers import T5Tokenizer, T5EncoderModel  # lazy import
            prot_tokenizer = T5Tokenizer.from_pretrained(net_name, trust_remote_code=True)
            prot_model = T5EncoderModel.from_pretrained(net_name, trust_remote_code=True)

        # ==== 5. ESMC (via the esm SDK) ====
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
            # ESMC does not use an HF tokenizer, so set it to None (MyDataset checks use_hf_api).
            prot_tokenizer = None

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
            # AIDO does not use an HF tokenizer, so set it to None (MyDataset checks use_hf_api).
            prot_tokenizer = None

        # ==== 7. VenusPLM ====
        elif net_name in ['AI4Protein/VenusPLM-300M']:
            from vplm import TransformerConfig, TransformerForMaskedLM, VPLMTokenizer

            cfg = TransformerConfig.from_pretrained(net_name, attn_impl="sdpa")
            prot_model = TransformerForMaskedLM.from_pretrained(net_name, config=cfg)
            prot_tokenizer = VPLMTokenizer.from_pretrained(net_name)
            use_hf_api = True

        # ==== 8. Full ProGen2 family (including local and HF paths) ====
        elif (net_name in [
            'hugohrban/progen2-small', 'hugohrban/progen2-medium', 'hugohrban/progen2-large',
            'hugohrban/progen2-base', 'hugohrban/progen2-xlarge', 'hugohrban/progen2-oas',
            'hugohrban/progen2-BFD90', 'hugohrban/progen2-small-mix7', 'hugohrban/progen2-small-mix7-bidi'
        ] or (os.path.exists(str(net_name)) and os.path.isdir(str(net_name)) and
              os.path.exists(os.path.join(str(net_name), 'config.json')))):
            # Support both Hugging Face model names and local checkpoint directories.

            # Check whether this is a local ProGen model (for example MAGE) and register custom classes if needed.
            is_local_progen = os.path.exists(str(net_name)) and os.path.isdir(str(net_name))
            if is_local_progen:
                config_path = os.path.join(str(net_name), 'config.json')
                if os.path.exists(config_path):
                    import json
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)

                    # Register custom modules when the checkpoint is a ProGen model with local implementation files.
                    if config_data.get('model_type') == 'progen':
                        modeling_file = os.path.join(str(net_name), 'modeling_progen.py')
                        config_file = os.path.join(str(net_name), 'configuration_progen.py')

                        if os.path.exists(modeling_file) and os.path.exists(config_file):
                            import sys
                            import importlib.util

                            # Add the model directory to sys.path so relative imports work.
                            model_dir = os.path.abspath(str(net_name))
                            if model_dir not in sys.path:
                                sys.path.insert(0, model_dir)

                            # Dynamically load the config module.
                            spec_config = importlib.util.spec_from_file_location("configuration_progen", config_file)
                            progen_config_module = importlib.util.module_from_spec(spec_config)
                            sys.modules['configuration_progen'] = progen_config_module
                            spec_config.loader.exec_module(progen_config_module)

                            # Dynamically load the model module.
                            spec_model = importlib.util.spec_from_file_location("modeling_progen", modeling_file)
                            progen_model_module = importlib.util.module_from_spec(spec_model)
                            sys.modules['modeling_progen'] = progen_model_module
                            spec_model.loader.exec_module(progen_model_module)

                            # Register the custom model implementation.
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

        # ==== 9. Ankh base / large (load directly from local paths) ====
        elif net_name in ['ankh-base', 'ankh-large']:
            print(f"Loading Ankh model from local path: {net_name}")
            from transformers import T5EncoderModel

            # Local path mapping.
            local_path_map = {
                'ankh-base': os.environ.get('ANKH_BASE_DIR', 'ankh-base'),
                'ankh-large': os.environ.get('ANKH_LARGE_DIR', 'ankh-large')
            }
            local_path = local_path_map[net_name]

            prot_tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
            prot_model = T5EncoderModel.from_pretrained(local_path, trust_remote_code=True)
            prot_model.eval()
            use_hf_api = True

        # ==== 10. Ankh3 large / xl (load from Hugging Face cache) ====
        elif net_name in ['ankh3-large', 'ankh3-xl']:
            print(f"Loading Ankh3 model from cache: {net_name}")
            from transformers import T5Tokenizer, T5EncoderModel

            # Hugging Face repo ID mapping.
            hf_model_name_map = {
                'ankh3-large': 'ElnaggarLab/ankh3-large',
                'ankh3-xl': 'ElnaggarLab/ankh3-xl'
            }
            hf_model_name = hf_model_name_map[net_name]

            # Use the Hugging Face repo ID so transformers loads from cache automatically.
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

            # Keep the local-cache optimization path (optional, does not affect online loading).
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
                            print(f"Loading ESM3 model: {model_name}")
                            self.client = ESM3.from_pretrained(model_name).to(device)
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
                    return torch.cat(outs, dim=0)

            prot_model = Esm3Model(model_name='esm3-sm-open-v1', max_length=MAX_LENGTH_AB)
            use_hf_api = False
            # ESM3 does not use an HF tokenizer, so set it to None (MyDataset checks use_hf_api).
            prot_tokenizer = None

        # ==== 13. Tranception ====
        elif net_name in [
            'OATML-Markslab/Tranception_Large',
            'OATML-Markslab/Tranception_Medium',
            'OATML-Markslab/Tranception_Small'
        ]:
            print(f"Loading Tranception model: {net_name}")
            prot_tokenizer = AutoTokenizer.from_pretrained(net_name, trust_remote_code=True)
            prot_model = AutoModel.from_pretrained(net_name, trust_remote_code=True)
            prot_model.eval()
            use_hf_api = True

        # ==== 14. Unsupported names ====
        else:
            raise ValueError(f"Unsupported model name: {net_name}")

        # ------------ Common finalization logic ------------
        # Ensure HF models are in eval() mode.
        try:
            if use_hf_api and hasattr(prot_model, 'eval'):
                prot_model.eval()
        except Exception:
            pass

        # Log max_position_embeddings and related metadata.
        try:
            _max_pos = getattr(prot_model, 'config', None)
            _max_pos = getattr(_max_pos, 'max_position_embeddings', None)
            logger.info(f"[model] max_position_embeddings={_max_pos}")
        except Exception:
            pass

    finally:
        # Do not clean up the tokenizer or model during inference; the caller still needs them.
        pass

    # Return the tokenizer, model, and the use_hf_api flag.
    return prot_tokenizer, prot_model, use_hf_api

# ============================================================================
# TokenCache class definition (copied from train.py)
# ============================================================================

from pathlib import Path as _Path

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


# Disable token caching during inference (optional optimization hook).
TOKEN_CACHE_DIR = None
TOKEN_CACHE_READONLY = False
TOKEN_CACHE = TokenCache(TOKEN_CACHE_DIR, readonly=TOKEN_CACHE_READONLY)

# ============================================================================
# MyDataset class definition (copied from train.py)
# ============================================================================

class MyDataset(Dataset):
    def __init__(self, data, prot_tokenizer, max_length_ab, max_length_ag, contain_label=True, use_hf_api=True, token_style: str = 'space', sep_token: str = '[SEP]', token_cache: Optional[TokenCache] = None):
        self.data = data
        self.prot_tokenizer = prot_tokenizer
        self.max_length_ab = max_length_ab
        self.max_length_ag = max_length_ag
        self.contain_label = contain_label
        self.use_hf_api = use_hf_api
        # tokenization behavior hints
        self.token_style = token_style if token_style in ('space', 'raw', 'auto') else 'space'
        self.sep_token = str(sep_token) if sep_token else '[SEP]'
        # Simple tokenization cache to avoid repeated encoding overhead.
        self._tok_cache = {}
        self.token_cache = token_cache

        # Auto-detect fallback for SaProt-like tokenizers: start with raw, switch to space if suspicious token counts
        if self.use_hf_api and self.prot_tokenizer is not None and self.token_style == 'auto':
            try:
                # Build a small sample of AB/AG in raw mode
                import pandas as _pd
                df = self.data.head(min(32, len(self.data))).reset_index(drop=True)
                # Construct sequences
                ab_texts = []
                ag_texts = []
                for _, row in df.iterrows():
                    hc = str(row.get('HC', '') or '')
                    lc = str(row.get('LC', '') or '')
                    ab = f"{hc}|{lc}" if (hc or lc) else str(row.get('Antibody', '') or '')
                    # raw mode replacement of uncommon AAs
                    ab_texts.append(re.sub(r"[UZOB]", "X", ab))
                    ag_texts.append(re.sub(r"[UZOB]", "X", str(row.get('Antigen', '') or '')))
                # Tokenize
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
        # Token length/truncation diagnostics (small sample)
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
                        ab_texts.append(re.sub(r"[UZOB]", "X", ab))
                        ag_texts.append(re.sub(r"[UZOB]", "X", str(row.get('Antigen', '') or '')))
                    else:
                        ab_texts.append(" ".join(list(re.sub(r"[UZOB]", "X", ab))))
                        ag_texts.append(" ".join(list(re.sub(r"[UZOB]", "X", str(row.get('Antigen', '') or '')))))

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

                ab_lens = [l for l in ( _tok_len(s) for s in ab_texts ) if l is not None]
                ag_lens = [l for l in ( _tok_len(s) for s in ag_texts ) if l is not None]
                if ab_lens and ag_lens:
                    ab_tr = sum(1 for l in ab_lens if l > int(self.max_length_ab)) / float(len(ab_lens))
                    ag_tr = sum(1 for l in ag_lens if l > int(self.max_length_ag)) / float(len(ag_lens))
                    logger.info(f"[tok] trunc_ratio_ab={ab_tr:.3f}, trunc_ratio_ag={ag_tr:.3f}, mean_len_ab={_stats.mean(ab_lens):.1f}, mean_len_ag={_stats.mean(ag_lens):.1f}, max_ab={self.max_length_ab}, max_ag={self.max_length_ag}")
        except Exception as _e:
            logger.debug(f"[tok] truncation diagnostics skipped: {_e}")

    @staticmethod
    def _clean_sequence(seq: str) -> str:
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
        if not text or text.lower() == "nan":
            return None
        return text

    def _extract_chain_texts(self, sample) -> Tuple[Optional[str], Optional[str]]:
        hc = self._optional_chain_text(sample.get("HC", None))
        lc = self._optional_chain_text(sample.get("LC", None))
        antibody = self._optional_chain_text(sample.get("Antibody", None))

        if antibody and (hc is None or lc is None):
            if "|" in antibody:
                hc_part, lc_part = antibody.split("|", 1)
                if hc is None:
                    hc = self._optional_chain_text(hc_part)
                if lc is None:
                    lc = self._optional_chain_text(lc_part)
            elif hc is None and lc is None:
                hc = antibody

        return hc, lc

    def _prepare_chain_sequence(self, chain_text: Optional[str]):
        if chain_text is None:
            return "" if self.token_style == "raw" else []
        if self.token_style == "raw":
            return chain_text
        return list(chain_text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        hc_text, lc_text = self._extract_chain_texts(sample)
        hc_present = torch.tensor(hc_text is not None, dtype=torch.bool)
        lc_present = torch.tensor(lc_text is not None, dtype=torch.bool)

        hc_input = self.tokenize_sequence(self._prepare_chain_sequence(hc_text), self.max_length_ab // 2)
        lc_input = self.tokenize_sequence(self._prepare_chain_sequence(lc_text), self.max_length_ab // 2)
        ag_input = self.tokenize_sequence(sample["Antigen"], self.max_length_ag)

        if self.contain_label:
            affinity = torch.tensor(sample["affinity"], dtype=torch.float32)
            return {
                "hc_input": hc_input,
                "lc_input": lc_input,
                "hc_present": hc_present,
                "lc_present": lc_present,
                "ag_input": ag_input,
                "affinity": affinity,
                "idx": index,
            }
        return {
            "hc_input": hc_input,
            "lc_input": lc_input,
            "hc_present": hc_present,
            "lc_present": lc_present,
            "ag_input": ag_input,
            "idx": index,
        }

    def pad_or_truncate(self, tensor, target_len=200):
        # Get the current tensor shape.
        current_len = tensor.shape[0]
        if current_len >= target_len:
            # Truncate when the current length exceeds the target length.
            return tensor[:target_len]
        else:
            # Pad when the current length is shorter than the target length.
            # Compute the shape of the padding block.
            padding_shape = [target_len - current_len] + list(tensor.shape[1:])
            padding = torch.zeros(padding_shape, dtype=tensor.dtype)
            return torch.cat([tensor, padding], dim=0)

    def tokenize_sequence(self, sequence, max_length):
        if self.use_hf_api:
            # Raw vs space-separated styles
            if self.token_style == 'raw':
                # Build a continuous string, preserving known separators
                if isinstance(sequence, (list, tuple)):
                    parts = []
                    for t in sequence:
                        t = str(t)
                        if len(t) == 1:
                            parts.append(re.sub(r"[UZOB]", "X", t))
                        else:
                            parts.append(t)
                    seq_text = "".join(parts)
                else:
                    seq_text = re.sub(r"[UZOB]", "X", str(sequence))
            else:
                # Space-separated, keep special sep token if present
                if isinstance(sequence, (list, tuple)):
                    tokens = []
                    for t in sequence:
                        t = str(t)
                        if len(t) == 1:
                            tokens.append(re.sub(r"[UZOB]", "X", t))
                        else:
                            tokens.append(t)
                    seq_text = " ".join(tokens)
                else:
                    seq_text = " ".join(re.sub(r"[UZOB]", "X", str(sequence)))

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
                # For current collate_fn we always need fixed shapes; if user disabled pad-to-max, warn once and still use max_length
                try:
                    if not getattr(args, 'pad_to_max', True):
                        global _PAD_TO_MAX_DISABLED_WARNED
                        try:
                            _ = _PAD_TO_MAX_DISABLED_WARNED
                        except NameError:
                            _PAD_TO_MAX_DISABLED_WARNED = False
                        if not _PAD_TO_MAX_DISABLED_WARNED:
                            logger.warning("--pad-to-max=False requested, but fixed padding is required by current collate. Falling back to 'max_length'.")
                            _PAD_TO_MAX_DISABLED_WARNED = True
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
            

# ============================================================================
# Net01 class definition (copied from train.py)
# ============================================================================

class Net01(nn.Module):
    def __init__(
        self,
        prot_model,
        prot_tokenizer,
        max_length_ab,
        max_length_ag,
        batch_size,
        # embed_dim=1024,
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

        # Freeze pretrained model parameters.
        if self.use_hf_api:
            # Check whether the model exposes parameters() (special models such as AIDO may not).
            if hasattr(self.prot_model, 'parameters') and callable(getattr(self.prot_model, 'parameters')):
                for param in self.prot_model.parameters():
                    param.requires_grad = False

        # Precompute delimiter token ids ('|') for masking in pooling (HF path)
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

        input_dim = 3 * self.embed_dim
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
            # IMPORTANT: do NOT remove explicit chain delimiter '|' by default
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
            # Reshape inputs from [B, 1, L] to [B, L].
            sequence_input_ids = sequence_input_ids.squeeze(1)
            sequence_attention_mask = sequence_attention_mask.squeeze(1)

            # Forward-pass keyword arguments.
            forward_kwargs = {
                "input_ids": sequence_input_ids,
                "attention_mask": sequence_attention_mask,
                "output_hidden_states": True,
            }

            # Check whether the model supports return_dict and ss_input_ids.
            _expects_ss = False
            try:
                _sig = signature(self.prot_model.forward)
                # Only add return_dict if the model explicitly supports it.
                if "return_dict" in _sig.parameters:
                    forward_kwargs["return_dict"] = True
                _expects_ss = ("ss_input_ids" in _sig.parameters)
            except Exception:
                # If the signature cannot be inspected, skip return_dict for safety.
                pass

            _cfg = getattr(self.prot_model, "config", None)
            _cfg_ss_vocab = int(getattr(_cfg, "ss_vocab_size", 0)) if _cfg is not None else 0
            if _expects_ss or _cfg_ss_vocab > 0:
                ss_input_ids = torch.zeros_like(sequence_input_ids, dtype=torch.long, device=sequence_input_ids.device)
                forward_kwargs["ss_input_ids"] = ss_input_ids

            # Run the model forward pass (with optional AMP).
            with torch.inference_mode():
                if device.type == 'cuda' and AMP_ENABLED:
                    with torch.cuda.amp.autocast(enabled=True, dtype=AMP_DTYPE):
                        sequence_output = self.prot_model(**forward_kwargs)
                else:
                    sequence_output = self.prot_model(**forward_kwargs)

            # Read the final hidden representation (compatible with Encoder/CausalLM outputs).
            if hasattr(sequence_output, 'last_hidden_state') and sequence_output.last_hidden_state is not None:
                emb = sequence_output.last_hidden_state.detach()
            elif hasattr(sequence_output, 'hidden_states') and sequence_output.hidden_states is not None:
                emb = sequence_output.hidden_states[-1].detach()
            else:
                if hasattr(sequence_output, 'logits') and sequence_output.logits is not None:
                    emb = sequence_output.logits.detach()
                else:
                    raise RuntimeError("Model output lacks last_hidden_state/hidden_states/logits; cannot extract embeddings")
            # Some providers (e.g. VenusPLM via vplm) return CPU tensors even when the module lives on CUDA.
            # Always align the embedding with the caller's device before downstream pooling.
            emb = emb.to(sequence_input_ids.device, non_blocking=True)
            # Align dimensions to [B, L, C].
            if emb.dim() != 3:
                raise RuntimeError(f"Expected 3D embedding, got {emb.dim()}D")
            B, L = sequence_input_ids.shape[:2]
            if emb.shape[0] == B and emb.shape[1] == L:
                pass  # [B,L,C]
            elif emb.shape[0] == L and emb.shape[1] == B:
                emb = emb.permute(1, 0, 2)
            # else: assume already [B,L,C] if mismatched we let it raise later
            sequence_embedding = emb
        else:
            with torch.inference_mode():
                sequence_embedding = self.prot_model(sequence_input_ids).detach()

        # Return [B, C, L].
        sequence_output = sequence_embedding.permute(0, 2, 1)
        return sequence_output

    def _infer_embed_dim(self) -> int:
        """Infer embedding dimension for backbone outputs."""
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
    # Tranception variants (MSA + Language Model)
    'tranception_large_01',
    'tranception_medium_01',
    'tranception_small_01',
 ]

net_names = [
    'Rostlab/prot_bert',
    'Rostlab/prot_bert_bfd',
    'Rostlab/prot_t5_xl_uniref50',
    'facebook/esm2_t30_150M_UR50D',
    'facebook/esm2_t33_650M_UR50D',
    'facebook/esm2_t36_3B_UR50D',
    # ESM-1v variants (5 models with different random seeds for ensemble predictions)
    'facebook/esm1v_t33_650M_UR90S_1',
    'facebook/esm1v_t33_650M_UR90S_2',
    'facebook/esm1v_t33_650M_UR90S_3',
    'facebook/esm1v_t33_650M_UR90S_4',
    'facebook/esm1v_t33_650M_UR90S_5',
    'EvolutionaryScale/esmc-300m-2024-12',
    'EvolutionaryScale/esmc-600m-2024-12',
    # ESM3 - EvolutionaryScale's latest generative protein model
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
    'proteinglm/proteinglm-3b-mlm',
    'proteinglm/proteinglm-3b-clm',
    'biomap-research/proteinglm-3b-mlm',
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
    # Tranception variants (MSA + Language Model for fitness prediction)
    'OATML-Markslab/Tranception_Large',
    'OATML-Markslab/Tranception_Medium',
    'OATML-Markslab/Tranception_Small',
 ]
