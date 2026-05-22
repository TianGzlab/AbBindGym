import numpy as np
import torch
from extra_utils import res_to_list, res_to_seq

class AbScores:
    def __init__(self, device='cpu', ncpu=1):
        self.device = device
        self.ncpu = ncpu
        
    def _initiate_abencoding(self, model, tokenizer):
        self.AbLang = model
        self.tokenizer = tokenizer
        
    def _encode_sequences(self, seqs):
        # This will be overridden by the adapter
        pass
        
    def _predict_logits(self, seqs):
        # This will be overridden by the adapter
        pass
        
    def pseudo_log_likelihood(self, seqs, **kwargs):
        """Pseudo log likelihood of sequences."""
        pass
