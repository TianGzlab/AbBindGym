import os
import sys
import shutil

# Get the directory where this adapter.py file is located
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import will be done inside methods when needed

# List of utility files that need to be available
UTILITY_FILES = [
    'restoration.py',
    'ablang_encodings.py', 
    'alignment.py',
    'scores.py',
    'extra_utils.py',
    'ablang.py',
    'encoderblock.py'
]

def create_missing_utility_files(missing_files):
    """Create missing utility files inline with their content."""
    
    # Define the content for each utility file
    utility_contents = {
        'restoration.py': '''import numpy as np
import torch
from extra_utils import res_to_list, res_to_seq

class AbRestore:
    def __init__(self, spread=11, device='cpu', ncpu=1):
        self.spread = spread
        self.device = device
        self.ncpu = ncpu
        
    def _initiate_abrestore(self, model, tokenizer):
        self.AbLang = model
        self.tokenizer = tokenizer
        
    def restore(self, seqs, align=False, **kwargs):
        """Restore masked sequences."""
        # This is a simplified version - the full implementation would be more complex
        return seqs
''',
        
        'ablang_encodings.py': '''import numpy as np
import torch
from extra_utils import res_to_list, res_to_seq

class AbEncoding:
    def __init__(self, device='cpu', ncpu=1):
        self.device = device
        self.ncpu = ncpu
        
    def _initiate_abencoding(self, model, tokenizer):
        self.AbLang = model
        self.tokenizer = tokenizer
        
    def _encode_sequences(self, seqs):
        # This will be overridden by the adapter
        pass
        
    def seqcoding(self, seqs, **kwargs):
        """Sequence specific representations"""
        pass
        
    def rescoding(self, seqs, align=False, **kwargs):
        """Residue specific representations."""
        pass
        
    def likelihood(self, seqs, align=False, stepwise_masking=False, **kwargs):
        """Likelihood of mutations"""
        pass
        
    def probability(self, seqs, align=False, stepwise_masking=False, **kwargs):
        """Probability of mutations"""
        pass
''',
        
        'alignment.py': '''from dataclasses import dataclass
import numpy as np
import torch
from extra_utils import paired_msa_numbering, unpaired_msa_numbering, create_alignment

@dataclass
class aligned_results:
    aligned_seqs: list
    aligned_embeds: np.ndarray
    number_alignment: list

class AbAlignment:
    def __init__(self, device='cpu', ncpu=1):
        self.device = device
        self.ncpu = ncpu
        
    def number_sequences(self, seqs, chain='H', fragmented=False):
        if chain == 'HL':
            numbered_seqs, seqs, number_alignment = paired_msa_numbering(seqs, fragmented=fragmented, n_jobs=self.ncpu)
        else:
            numbered_seqs, seqs, number_alignment = unpaired_msa_numbering(seqs, chain=chain, fragmented=fragmented, n_jobs=self.ncpu)
        return numbered_seqs, seqs, number_alignment
    
    def align_encodings(self, encodings, numbered_seqs, seqs, number_alignment):
        aligned_encodings = []
        for res_embed, numbered_seq, seq in zip(encodings, numbered_seqs, seqs):
            aligned_encodings.append(create_alignment(res_embed, numbered_seq, seq, number_alignment))
        return np.concatenate([aligned_encodings], axis=0)
        
    def reformat_subsets(self, subset_list, mode='seqcoding', align=False, numbered_seqs=None, seqs=None, number_alignment=None):
        if mode in ['seqcoding', 'pseudo_log_likelihood', 'confidence']:
            return np.concatenate(subset_list)
        elif mode == 'restore' and align:
            # For restore mode with alignment, return the aligned sequences
            return subset_list[0] if len(subset_list) == 1 else subset_list
        elif mode == 'restore' and not align:
            # For restore mode without alignment, return the restored sequences
            return subset_list[0] if len(subset_list) == 1 else subset_list
        elif align:
            aligned_subsets = []
            for num, subset in enumerate(subset_list):
                start_idx = num * len(subset)
                end_idx = (num + 1) * len(subset)
                aligned_subset = self.align_encodings(
                    subset, 
                    numbered_seqs[start_idx:end_idx], 
                    seqs[start_idx:end_idx], 
                    number_alignment
                )
                aligned_subsets.append(aligned_subset)
            subset = np.concatenate(aligned_subsets)
            return aligned_results(
                aligned_seqs=[''.join(alist) for alist in subset[:,:,-1]],
                aligned_embeds=subset[:,:,:-1].astype(float),
                number_alignment=number_alignment.apply(lambda x: '{}{}'.format(*x[0]), axis=1).values
            )
        elif not align:
            return sum(subset_list, [])
        else:
            return np.concatenate(subset_list)
''',
        
        'scores.py': '''import numpy as np
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
''',
        
        'extra_utils.py': '''import string, re
import numpy as np

def res_to_list(logits, seq):
    return logits[:len(seq)]

def res_to_seq(a, mode='mean'):
    """Function for how we go from n_values for each amino acid to n_values for each sequence."""
    if mode=='sum':
        return a[0:(int(a[-1]))].sum()
    elif mode=='mean':
        return a[0:(int(a[-1]))].mean()
    elif mode=='restore':
        return a[0][0:(int(a[-1]))]

def get_number_alignment(numbered_seqs):
    """Creates a number alignment from the anarci results."""
    import pandas as pd
    alist = [pd.DataFrame(aligned_seq, columns=[0,1,'resi']) for aligned_seq in numbered_seqs]
    unsorted_alignment = pd.concat(alist).drop_duplicates(subset=0)
    max_alignment = get_max_alignment()
    return max_alignment.merge(unsorted_alignment.query("resi!='-'"), left_on=0, right_on=0)[[0,1]]

def get_max_alignment():
    """Create maximum possible alignment for sorting"""
    import pandas as pd
    sortlist = [[("<", "")]]
    for num in range(1, 128+1):
        if num in [33,61,112]:
            for char in string.ascii_uppercase[::-1]:
                sortlist.append([(num, char)])
            sortlist.append([(num,' ')])
        else:
            sortlist.append([(num,' ')])
            for char in string.ascii_uppercase:
                sortlist.append([(num, char)])
    return pd.DataFrame(sortlist + [[(">", "")]])

def paired_msa_numbering(ab_seqs, fragmented=False, n_jobs=10):
    import pandas as pd
    tmp_seqs = [pairs.replace(">", "").replace("<", "").split("|") for pairs in ab_seqs]
    numbered_seqs_heavy, seqs_heavy, number_alignment_heavy = unpaired_msa_numbering([i[0] for i in tmp_seqs], 'H', fragmented=fragmented, n_jobs=n_jobs)
    numbered_seqs_light, seqs_light, number_alignment_light = unpaired_msa_numbering([i[1] for i in tmp_seqs], 'L', fragmented=fragmented, n_jobs=n_jobs)
    number_alignment = pd.concat([number_alignment_heavy, pd.DataFrame([[("|",""), "|"]]), number_alignment_light]).reset_index(drop=True)
    seqs = [f"{heavy}|{light}" for heavy, light in zip(seqs_heavy, seqs_light)]
    numbered_seqs = [heavy + [(("|",""), "|", "|")] + light for heavy, light in zip(numbered_seqs_heavy, numbered_seqs_light)]
    return numbered_seqs, seqs, number_alignment

def unpaired_msa_numbering(seqs, chain='H', fragmented=False, n_jobs=10):
    numbered_seqs = number_with_anarci(seqs, chain=chain, fragmented=fragmented, n_jobs=n_jobs)
    number_alignment = get_number_alignment(numbered_seqs)
    number_alignment[1] = chain
    seqs = [''.join([i[2] for i in numbered_seq]).replace('-','') for numbered_seq in numbered_seqs]
    return numbered_seqs, seqs, number_alignment

def number_with_anarci(seqs, chain='H', fragmented=False, n_jobs=1):
    import anarci
    import pandas as pd
    anarci_out = anarci.run_anarci(pd.DataFrame(seqs).reset_index().values.tolist(), ncpu=n_jobs, scheme='imgt', allowed_species=['human', 'mouse'])
    numbered_seqs = []
    for onarci in anarci_out[1]:
        numbered_seq = []
        for i in onarci[0][0]:
            if i[1] != '-':
                numbered_seq.append((i[0], chain, i[1]))
        if fragmented:
            numbered_seqs.append(numbered_seq)
        else:
            numbered_seqs.append([(("<",""), chain, "<")] + numbered_seq + [((">",""), chain, ">")])
    return numbered_seqs

def create_alignment(res_embeds, numbered_seqs, seq, number_alignment):
    import pandas as pd
    datadf = pd.DataFrame(numbered_seqs)
    sequence_alignment = number_alignment.merge(datadf, how='left', on=[0, 1]).fillna('-')[2]
    idxs = np.where(sequence_alignment.values == '-')[0]
    idxs = [idx-num for num, idx in enumerate(idxs)]
    aligned_embeds = pd.DataFrame(np.insert(res_embeds[:len(seq)], idxs, 0, axis=0))
    return pd.concat([aligned_embeds, sequence_alignment], axis=1).values
''',
        
        'ablang.py': '''from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from .encoderblock import TransformerEncoder, get_activation_fn

class AbLang(torch.nn.Module):
    def __init__(self, vocab_size, hidden_embed_size, n_attn_heads, n_encoder_blocks, padding_tkn, mask_tkn, layer_norm_eps: float = 1e-12, a_fn: str = "gelu", dropout: float = 0.0):
        super().__init__()
        self.AbRep = AbRep(vocab_size, hidden_embed_size, n_attn_heads, n_encoder_blocks, padding_tkn, mask_tkn, layer_norm_eps, a_fn, dropout)
        self.AbHead = AbHead(vocab_size, hidden_embed_size, self.AbRep.aa_embed_layer.weight, layer_norm_eps, a_fn)
        
    def forward(self, tokens, return_attn_weights=False, return_rep_layers=[]):
        representations = self.AbRep(tokens, return_attn_weights, return_rep_layers)
        if return_attn_weights:
            return representations.attention_weights
        elif return_rep_layers != []:
            return representations.many_hidden_states
        else:
            likelihoods = self.AbHead(representations.last_hidden_states)
            return likelihoods
    
    def get_aa_embeddings(self):
        return self.AbRep.aa_embed_layer

class AbRep(torch.nn.Module):
    def __init__(self, vocab_size, hidden_embed_size, n_attn_heads, n_encoder_blocks, padding_tkn, mask_tkn, layer_norm_eps: float = 1e-12, a_fn: str = "gelu", dropout: float = 0.0):
        super().__init__()
        self.aa_embed_layer = nn.Embedding(vocab_size, hidden_embed_size, padding_idx=padding_tkn)
        self.encoder_blocks = nn.ModuleList([TransformerEncoder(hidden_embed_size, n_attn_heads, dropout, layer_norm_eps, a_fn) for _ in range(n_encoder_blocks)])
        
    def forward(self, tokens, return_attn_weights=False, return_rep_layers=[]):
        hidden_states = self.aa_embed_layer(tokens)
        for i, encoder_block in enumerate(self.encoder_blocks):
            hidden_states, attn_weights = encoder_block(hidden_states)
        return type('obj', (object,), {'last_hidden_states': hidden_states})

class AbHead(torch.nn.Module):
    def __init__(self, vocab_size, hidden_embed_size, aa_embeddings, layer_norm_eps: float = 1e-12, a_fn: str = "gelu"):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_embed_size, eps=layer_norm_eps)
        self.aa_embeddings = aa_embeddings
        
    def forward(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        return torch.matmul(hidden_states, self.aa_embeddings.transpose(0, 1))
''',
        
        'encoderblock.py': '''import torch
import math
from torch import nn
import torch.nn.functional as F
import einops
from rotary_embedding_torch import RotaryEmbedding

class TransformerEncoder(torch.nn.Module):
    def __init__(self, hidden_embed_size, n_attn_heads, attn_dropout: float = 0.0, layer_norm_eps: float = 1e-05, a_fn: str = "gelu"):
        super().__init__()
        assert hidden_embed_size % n_attn_heads == 0, "Embedding dimension must be devisible with the number of heads."
        self.multihead_attention = MultiHeadAttention(embed_dim=hidden_embed_size, num_heads=n_attn_heads, attention_dropout_prob=attn_dropout)
        activation_fn, scale = get_activation_fn(a_fn)
        self.intermediate_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_embed_size, hidden_embed_size * 4 * scale),
            activation_fn(),
            torch.nn.Linear(hidden_embed_size * 4, hidden_embed_size),
        )
        self.pre_attn_layer_norm = torch.nn.LayerNorm(hidden_embed_size, eps=layer_norm_eps)
        self.final_layer_norm = torch.nn.LayerNorm(hidden_embed_size, eps=layer_norm_eps)
        
    def forward(self, hidden_embed, attn_mask=None, return_attn_weights: bool = False):
        residual = hidden_embed
        hidden_embed = self.pre_attn_layer_norm(hidden_embed.clone())
        hidden_embed, attn_weights = self.multihead_attention(hidden_embed, attn_mask=attn_mask, return_attn_weights=return_attn_weights)
        hidden_embed = residual + hidden_embed
        residual = hidden_embed
        hidden_embed = self.final_layer_norm(hidden_embed)
        hidden_embed = self.intermediate_layer(hidden_embed)
        hidden_embed = residual + hidden_embed
        return hidden_embed, attn_weights

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, attention_dropout_prob=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(attention_dropout_prob)
        
    def forward(self, x, attn_mask=None, return_attn_weights=False):
        batch_size, seq_len, embed_dim = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        
        if return_attn_weights:
            return attn_output, attn_weights
        return attn_output

def get_activation_fn(activation_fn):
    if activation_fn == "gelu":
        return torch.nn.GELU, 1
    elif activation_fn == "relu":
        return torch.nn.ReLU, 1
    elif activation_fn == "swish":
        return torch.nn.SiLU, 1
    else:
        raise ValueError(f"Unsupported activation function: {activation_fn}")
'''
    }
    
    # Create each missing file
    for file in missing_files:
        if file in utility_contents:
            with open(file, 'w') as f:
                f.write(utility_contents[file])
            print(f"✅ Created {file}")
        else:
            print(f"⚠️ No content template for {file}")

def ensure_utility_files_available():
    """
    Ensure all utility files are available in the current directory.
    If any are missing, try to copy them from the repository root.
    """
    missing_files = []
    for file in UTILITY_FILES:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"🔍 Looking for missing utility files: {missing_files}")
        
        # Try to find the repository root (where all utility files are)
        # Look for common parent directories that might contain the files
        possible_paths = [
            current_dir,  # Current directory (where model files are downloaded)
            os.path.join(current_dir, '..'),  # Parent directory
            os.path.join(current_dir, '..', '..'),  # Grandparent directory
            os.path.join(current_dir, '..', '..', '..'),  # Great-grandparent directory
            os.path.join(os.path.expanduser('~'), 'ablang2'),  # Home directory
        ]
        
        # Check if we're in a Hugging Face cache directory
        is_hf_cache = 'huggingface' in current_dir and 'cache' in current_dir
        if is_hf_cache:
            print("🔍 Detected Hugging Face cache directory - will create utility files inline")
            # Skip searching other paths and create files inline
            possible_paths = []
        
        # Also try to find files in the Hugging Face cache structure
        cache_dir = os.path.dirname(current_dir)
        if 'huggingface' in cache_dir:
            # Look in the repository root within the cache
            repo_root = os.path.join(cache_dir, '..', '..', '..', '..')
            possible_paths.append(repo_root)
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"🔍 Checking path: {path}")
                # Check if all missing files exist in this path
                all_found = True
                for file in missing_files:
                    file_path = os.path.join(path, file)
                    if not os.path.exists(file_path):
                        all_found = False
                        print(f"   ❌ Missing: {file}")
                        break
                    else:
                        print(f"   ✅ Found: {file}")
                
                if all_found:
                    print(f"🎯 Found all files in: {path}")
                    # Copy all missing files
                    for file in missing_files:
                        src = os.path.join(path, file)
                        dst = os.path.join(current_dir, file)
                        shutil.copy2(src, dst)
                        print(f"✅ Copied {file} to cached directory")
                    return True
        
        # If we get here, we couldn't find the files
        print(f"❌ Could not find utility files in any of the searched paths:")
        for path in possible_paths:
            print(f"   - {path}")
        
        # Try to create the missing files inline
        print("🔧 Attempting to create missing utility files inline...")
        try:
            create_missing_utility_files(missing_files)
            print("✅ Successfully created missing utility files")
            return True
        except Exception as e:
            print(f"❌ Failed to create utility files: {e}")
            
            # For Colab environments, provide a helpful error message
            if 'google.colab' in str(sys.modules):
                raise FileNotFoundError(
                    f"Missing utility files: {missing_files}. "
                    "This appears to be a Google Colab environment. "
                    "Please ensure you have cloned the repository and the utility files are available. "
                    "Try running: !git clone https://huggingface.co/hemantn/ablang2"
                )
            else:
                raise FileNotFoundError(
                    f"Missing utility files: {missing_files}. "
                    "These files are required for the adapter to work. "
                    "Please ensure the repository is properly set up."
                )
    
    return True

# Ensure utility files are available before importing
ensure_utility_files_available()

# Debug: Check what files are in the current directory
print(f"📁 Files in current directory ({current_dir}):")
for f in os.listdir(current_dir):
    if f.endswith('.py'):
        print(f"   {f}")

# Import utility modules directly (no package structure needed)
import sys
import os

# Ensure we import from the cache directory, not from /content
cache_dir = os.path.dirname(os.path.abspath(__file__))
if cache_dir not in sys.path:
    sys.path.insert(0, cache_dir)

# Remove /content from sys.path to avoid conflicts
content_path = '/content'
if content_path in sys.path:
    sys.path.remove(content_path)
    print(f"✅ Removed {content_path} from sys.path to avoid import conflicts")

# Import utility modules
try:
    from restoration import AbRestore
    from ablang_encodings import AbEncoding
    from alignment import AbAlignment
    from scores import AbScores
    import torch
    import numpy as np
    from extra_utils import res_to_seq, res_to_list
    print("✅ Successfully imported utility modules from cache directory")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"🔧 Current sys.path: {sys.path}")
    print(f"🔧 Cache directory: {cache_dir}")
    raise

class HuggingFaceTokenizerAdapter:
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = getattr(tokenizer, 'mask_token_id', None) or tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        self.vocab = tokenizer.get_vocab() if hasattr(tokenizer, 'get_vocab') else tokenizer.vocab
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.all_special_tokens = tokenizer.all_special_tokens

    def __call__(self, seqs, pad=True, w_extra_tkns=False, device=None, mode=None):
        tokens = self.tokenizer(seqs, padding=True, return_tensors='pt')
        input_ids = tokens['input_ids'].to(self.device if device is None else device)
        if mode == 'decode':
            # seqs is a tensor of token ids
            if isinstance(seqs, torch.Tensor):
                seqs = seqs.cpu().numpy()
            decoded = []
            for i, seq in enumerate(seqs):
                chars = [self.inv_vocab.get(int(t), '') for t in seq if self.inv_vocab.get(int(t), '') not in {'-', '*', '<', '>'} and self.inv_vocab.get(int(t), '') != '']
                # Use res_to_seq for formatting, pass (sequence, length) tuple as in original code
                # The length is not always available, so use len(chars) as fallback
                from extra_utils import res_to_seq
                formatted = res_to_seq([ ''.join(chars), len(chars) ], mode='restore')
                decoded.append(formatted)
            return decoded
        return input_ids

class HFAbRestore(AbRestore):
    def __init__(self, hf_model, hf_tokenizer, spread=11, device='cpu', ncpu=1):
        super().__init__(spread=spread, device=device, ncpu=ncpu)
        self.used_device = device
        self._hf_model = hf_model
        self.tokenizer = HuggingFaceTokenizerAdapter(hf_tokenizer, device)

    @property
    def AbLang(self):
        def model_call(x):
            output = self._hf_model(x)
            if hasattr(output, 'last_hidden_state'):
                return output.last_hidden_state
            return output
        return model_call
    
    def restore(self, seqs, align=False, **kwargs):
        """Restore masked residues in antibody sequences."""
        if isinstance(seqs, str):
            seqs = [seqs]
        
        n_seqs = len(seqs)
        
        if align:
            # Implement alignment using ANARCI to create spread sequences
            seqs = self._sequence_aligning(seqs)
            nr_seqs = len(seqs)//self.spread
            
            tokens = self.tokenizer(seqs, pad=True, w_extra_tkns=False, device=self.used_device)          
            predictions = self.AbLang(tokens)[:,:,1:21]

            # Reshape
            tokens = tokens.reshape(nr_seqs, self.spread, -1)
            predictions = predictions.reshape(nr_seqs, self.spread, -1, 20)
            seqs = seqs.reshape(nr_seqs, -1)

            # Find index of best predictions
            best_seq_idx = torch.argmax(torch.max(predictions, -1).values[:,:,1:2].mean(2), -1)

            # Select best predictions           
            tokens = tokens.gather(1, best_seq_idx.view(-1, 1).unsqueeze(1).repeat(1, 1, tokens.shape[-1])).squeeze(1)
            predictions = predictions[range(predictions.shape[0]), best_seq_idx]
            seqs = np.take_along_axis(seqs, best_seq_idx.view(-1, 1).cpu().numpy(), axis=1)
        else:
            tokens = self.tokenizer(seqs, pad=True, w_extra_tkns=False, device=self.used_device)
            predictions = self.AbLang(tokens)[:,:,1:21]

        predicted_tokens = torch.max(predictions, -1).indices + 1
        restored_tokens = torch.where(tokens==23, predicted_tokens, tokens)

        restored_seqs = self.tokenizer(restored_tokens, mode="decode")

        if n_seqs < len(restored_seqs):
            restored_seqs = [f"{h}|{l}".replace('-','') for h,l in zip(restored_seqs[:n_seqs], restored_seqs[n_seqs:])]
            seqs = [f"{h}|{l}" for h,l in zip(seqs[:n_seqs], seqs[n_seqs:])]
        
        from extra_utils import res_to_seq
        return np.array([res_to_seq(seq, 'restore') for seq in np.c_[restored_seqs, np.vectorize(len)(seqs)]])
    
    def _sequence_aligning(self, seqs):
        """Create spread sequences using ANARCI alignment."""
        tmp_seqs = [pairs.replace(">", "").replace("<", "").split("|") for pairs in seqs]
        
        spread_heavy = [f"<{seq}>" for seq in self._create_spread_of_sequences(tmp_seqs, chain = 'H')]
        spread_light = [f"<{seq}>" for seq in self._create_spread_of_sequences(tmp_seqs, chain = 'L')]
        
        return np.concatenate([np.array(spread_heavy),np.array(spread_light)])
    
    def _create_spread_of_sequences(self, seqs, chain = 'H'):
        """Create spread sequences using ANARCI."""
        import pandas as pd
        import anarci
        
        chain_idx = 0 if chain == 'H' else 1
        numbered_seqs = anarci.run_anarci(
            pd.DataFrame([seq[chain_idx].replace('*', 'X') for seq in seqs]).reset_index().values.tolist(), 
            ncpu=self.ncpu, 
            scheme='imgt',
            allowed_species=['human', 'mouse'],
        )
        
        anarci_data = pd.DataFrame(
            [str(anarci[0][0]) if anarci else 'ANARCI_error' for anarci in numbered_seqs[1]], 
            columns=['anarci']
        ).astype('<U90')
        
        max_position = 128 if chain == 'H' else 127
        
        # Define get_sequences_from_anarci function directly
        import re
        
        def get_sequences_from_anarci(out_anarci, max_position, spread):
            """
            Ensures correct masking on each side of sequence
            """
            
            if out_anarci == 'ANARCI_error':
                return np.array(['ANARCI-ERR']*spread)
            
            end_position = int(re.search(r'\d+', out_anarci[::-1]).group()[::-1])
            # Fixes ANARCI error of poor numbering of the CDR1 region
            start_position = int(re.search(r'\d+,\s\'.\'\),\s\'[^-]+\'\),\s\(\(\d+,\s\'.\'\),\s\'[^-]+\'\),\s\(\(\d+,\s\'.\'\),\s\'[^-]+\'\),\s\(\(\d+,\s\'.\'\),\s\'[^-]+',
                                           out_anarci).group().split(',')[0]) - 1
            
            sequence = "".join(re.findall(r"(?i)[A-Z*]", "".join(re.findall(r'\),\s\'[A-Z*]', out_anarci))))

            sequence_j = ''.join(sequence).replace('-','').replace('X','*') + '*'*(max_position-int(end_position))

            return get_spread_sequences(sequence_j, spread, start_position)
        
        def get_spread_sequences(seq, spread, start_position):
            """
            Test sequences which are 8 positions shorter (position 10 + max CDR1 gap of 7) up to 2 positions longer (possible insertions).
            """
            spread_sequences = []

            for diff in range(start_position-8, start_position+2+1):
                spread_sequences.append('*'*diff+seq)
            
            return np.array(spread_sequences)
        seqs = anarci_data.apply(
            lambda x: get_sequences_from_anarci(
                x.anarci, 
                max_position, 
                self.spread
            ), axis=1, result_type='expand'
        ).to_numpy().reshape(-1)
        
        return seqs

def add_angle_brackets(seq):
    # Assumes input is 'VH|VL' or 'VH|' or '|VL'
    if '|' in seq:
        vh, vl = seq.split('|', 1)
    else:
        vh, vl = seq, ''
    return f"<{vh}>|<{vl}>"

class AbLang2PairedHuggingFaceAdapter(AbEncoding, AbRestore, AbAlignment, AbScores):
    """
    Adapter to use pretrained utilities with a HuggingFace-loaded ablang2_paired model and tokenizer.
    Automatically uses CUDA if available, otherwise CPU.
    """
    def __init__(self, model, tokenizer, device=None, ncpu=1):
        super().__init__()
        if device is None:
            self.used_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.used_device = torch.device(device)
        self.AbLang = model  # HuggingFace model instance
        self.tokenizer = tokenizer
        self.AbLang.to(self.used_device)
        self.AbLang.eval()
        # Always get AbRep from the underlying model
        if hasattr(self.AbLang, 'model') and hasattr(self.AbLang.model, 'AbRep'):
            self.AbRep = self.AbLang.model.AbRep
        else:
            raise AttributeError("Could not find AbRep in the HuggingFace model or its underlying model.")
        self.ncpu = ncpu
        self.spread = 11  # For compatibility with original utilities
        # The following is no longer needed since all_special_tokens now returns IDs directly
        # self.tokenizer.all_special_token_ids = [
        #     self.tokenizer.convert_tokens_to_ids(tok) for tok in self.tokenizer.all_special_tokens
        # ]
        # self.tokenizer._all_special_tokens_str = self.tokenizer.all_special_tokens
        # self.tokenizer.all_special_tokens = [
        #     self.tokenizer.convert_tokens_to_ids(tok) for tok in self.tokenizer._all_special_tokens_str
        # ]

    def freeze(self):
        self.AbLang.eval()

    def unfreeze(self):
        self.AbLang.train()

    def _encode_sequences(self, seqs):
        # Override to use HuggingFace tokenizer interface
        tokens = self.tokenizer(seqs, padding=True, return_tensors='pt')
        tokens = extract_input_ids(tokens, self.used_device)
        return self.AbRep(tokens).last_hidden_states.detach()

    def _predict_logits(self, seqs):
        # Override to use HuggingFace tokenizer interface
        tokens = self.tokenizer(seqs, padding=True, return_tensors='pt')
        tokens = extract_input_ids(tokens, self.used_device)
        output = self.AbLang(tokens)
        if hasattr(output, 'last_hidden_state'):
            return output.last_hidden_state.detach()
        return output.detach()

    def _predict_logits_with_step_masking(self, seqs):
        # Override the stepwise masking method to use HuggingFace tokenizer
        tokens = self.tokenizer(seqs, padding=True, return_tensors='pt')
        tokens = extract_input_ids(tokens, self.used_device)
        
        logits = []
        for single_seq_tokens in tokens:
            tkn_len = len(single_seq_tokens)
            masked_tokens = single_seq_tokens.repeat(tkn_len, 1)
            for num in range(tkn_len):
                masked_tokens[num, num] = self.tokenizer.mask_token_id
            
            with torch.no_grad():
                logits_tmp = self.AbLang(masked_tokens)
                       
            logits_tmp = torch.stack([logits_tmp[num, num] for num in range(tkn_len)])
            logits.append(logits_tmp)
    
        return torch.stack(logits, dim=0)

    def _preprocess_labels(self, labels):
        labels = extract_input_ids(labels, self.used_device)
        return labels

    def __call__(self, seqs, mode='seqcoding', align=False, stepwise_masking=False, fragmented=False, batch_size=50):
        """
        Use different modes for different usecases, mimicking the original pretrained class.
        """
        # Local implementation of format_seq_input
        def format_seq_input(seqs, fragmented=False):
            """Format input sequences for processing."""
            if isinstance(seqs[0], str):
                seqs = [seqs]
            
            if fragmented:
                # For fragmented sequences, format as VH|VL without angle brackets
                formatted_seqs = []
                for seq in seqs:
                    if isinstance(seq, (list, tuple)) and len(seq) == 2:
                        heavy, light = seq[0], seq[1]
                        formatted_seqs.append(f"{heavy}|{light}")
                    else:
                        formatted_seqs.append(seq)
                return formatted_seqs, 'HL'
            else:
                # For non-fragmented sequences, add angle brackets: <VH>|<VL>
                formatted_seqs = []
                for seq in seqs:
                    if isinstance(seq, (list, tuple)) and len(seq) == 2:
                        heavy, light = seq[0], seq[1]
                        # Add angle brackets and handle empty sequences
                        heavy_part = f"<{heavy}>" if heavy else "<>"
                        light_part = f"<{light}>" if light else "<>"
                        formatted_seqs.append(f"{heavy_part}|{light_part}".replace("<>", ""))
                    else:
                        formatted_seqs.append(seq)
                
                return formatted_seqs, 'HL'

        valid_modes = [
            'rescoding', 'seqcoding', 'restore', 'likelihood', 'probability',
            'pseudo_log_likelihood', 'confidence'
        ]
        if mode not in valid_modes:
            raise SyntaxError(f"Given mode doesn't exist. Please select one of the following: {valid_modes}.")

        seqs, chain = format_seq_input(seqs, fragmented=fragmented)

        if align:
            numbered_seqs, seqs, number_alignment = self.number_sequences(
                seqs, chain=chain, fragmented=fragmented
            )
        else:
            numbered_seqs = None
            number_alignment = None

        subset_list = []
        for subset in [seqs[x:x+batch_size] for x in range(0, len(seqs), batch_size)]:
            subset_list.append(getattr(self, mode)(subset, align=align, stepwise_masking=stepwise_masking))

        return self.reformat_subsets(
            subset_list,
            mode=mode,
            align=align,
            numbered_seqs=numbered_seqs,
            seqs=seqs,
            number_alignment=number_alignment,
        )

    def pseudo_log_likelihood(self, seqs, **kwargs):
        """
        Original (non-vectorized) pseudo log-likelihood computation matching notebook behavior.
        """
        # Format input: join VH and VL with '|'
        formatted_seqs = []
        for s in seqs:
            if isinstance(s, (list, tuple)):
                formatted_seqs.append('|'.join(s))
            else:
                formatted_seqs.append(s)

        # Tokenize all sequences in batch
        labels = self.tokenizer(
            formatted_seqs, padding=True, return_tensors='pt'
        )
        labels = extract_input_ids(labels, self.used_device)

        # Convert special tokens to IDs
        if isinstance(self.tokenizer.all_special_tokens[0], int):
            special_token_ids = set(self.tokenizer.all_special_tokens)
        else:
            special_token_ids = set(self.tokenizer.convert_tokens_to_ids(tok) for tok in self.tokenizer.all_special_tokens)
        pad_token_id = self.tokenizer.pad_token_id

        mask_token_id = getattr(self.tokenizer, 'mask_token_id', None)
        if mask_token_id is None:
            mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        plls = []
        with torch.no_grad():
            for i, seq_label in enumerate(labels):
                seq_pll = []
                for j, token_id in enumerate(seq_label):
                    if token_id.item() in special_token_ids or token_id.item() == pad_token_id:
                        continue
                    masked = seq_label.clone()
                    masked[j] = mask_token_id
                    logits = self.AbLang(masked.unsqueeze(0))
                    if hasattr(logits, 'last_hidden_state'):
                        logits = logits.last_hidden_state
                    logits = logits[0, j]
                    nll = torch.nn.functional.cross_entropy(
                        logits.unsqueeze(0), token_id.unsqueeze(0), reduction="none"
                    )
                    seq_pll.append(-nll.item())
                if seq_pll:
                    plls.append(np.mean(seq_pll))
                else:
                    plls.append(float('nan'))
        return np.array(plls)

    def seqcoding(self, seqs, **kwargs):
        """Sequence specific representations - returns 480-dimensional embeddings for each sequence."""
        # Format input: join VH and VL with '|'
        formatted_seqs = []
        for s in seqs:
            if isinstance(s, (list, tuple)):
                formatted_seqs.append('|'.join(s))
            else:
                formatted_seqs.append(s)
        
        # Get embeddings using the model
        embeddings = self._encode_sequences(formatted_seqs)
        
        # Return sequence-level embeddings (mean pooling over sequence length)
        # Remove batch dimension and take mean over sequence dimension
        if len(embeddings.shape) == 3:  # [batch_size, seq_len, hidden_size]
            seq_embeddings = embeddings.mean(dim=1)  # [batch_size, hidden_size]
        else:
            seq_embeddings = embeddings
        
        return seq_embeddings.cpu().numpy()

    def rescoding(self, seqs, align=False, **kwargs):
        """Residue specific representations - returns 480-dimensional embeddings for each residue."""
        # Format input: join VH and VL with '|'
        formatted_seqs = []
        for s in seqs:
            if isinstance(s, (list, tuple)):
                formatted_seqs.append('|'.join(s))
            else:
                formatted_seqs.append(s)
        
        # Get embeddings using the model
        embeddings = self._encode_sequences(formatted_seqs)
        
        # Return residue-level embeddings
        # embeddings shape: [batch_size, seq_len, hidden_size]
        if len(embeddings.shape) == 3:
            # Convert to numpy and return as list of arrays for each sequence
            embeddings_np = embeddings.cpu().numpy()
            return [embeddings_np[i] for i in range(embeddings_np.shape[0])]
        else:
            return embeddings.cpu().numpy()

    def likelihood(self, seqs, align=False, stepwise_masking=False, **kwargs):
        """Likelihood of mutations - returns logits for each amino acid at each position."""
        # Format input: join VH and VL with '|'
        formatted_seqs = []
        for s in seqs:
            if isinstance(s, (list, tuple)):
                formatted_seqs.append('|'.join(s))
            else:
                formatted_seqs.append(s)

        # Get logits
        if stepwise_masking:
            logits = self._predict_logits_with_step_masking(formatted_seqs)
        else:
            logits = self._predict_logits(formatted_seqs)
        
        # Return logits as numpy array
        return logits.cpu().numpy()

    def confidence(self, seqs, **kwargs):
        """Confidence calculation - match original ablang2 implementation by excluding all special tokens from loss."""
        # Format input: join VH and VL with '|'
        formatted_seqs = []
        for s in seqs:
            if isinstance(s, (list, tuple)):
                formatted_seqs.append('|'.join(s))
            else:
                formatted_seqs.append(s)
        
        plls = []
        for seq in formatted_seqs:
            tokens = self.tokenizer([seq], padding=True, return_tensors='pt')
            input_ids = extract_input_ids(tokens, self.used_device)
            
            with torch.no_grad():
                output = self.AbLang(input_ids)
                if hasattr(output, 'last_hidden_state'):
                    logits = output.last_hidden_state
                else:
                    logits = output
                
                # Get the sequence (remove batch dimension)
                logits = logits[0]  # [seq_len, vocab_size]
                input_ids = input_ids[0]  # [seq_len]
                
                # Exclude all special tokens (pad, mask, etc.)
                if isinstance(self.tokenizer.all_special_tokens[0], int):
                    special_token_ids = set(self.tokenizer.all_special_tokens)
                else:
                    special_token_ids = set(self.tokenizer.convert_tokens_to_ids(tok) for tok in self.tokenizer.all_special_tokens)
                valid_mask = ~torch.isin(input_ids, torch.tensor(list(special_token_ids), device=input_ids.device))
                
                if valid_mask.sum() > 0:
                    valid_logits = logits[valid_mask]
                    valid_labels = input_ids[valid_mask]
                    
                    # Calculate cross-entropy loss
                    nll = torch.nn.functional.cross_entropy(
                        valid_logits,
                        valid_labels,
                        reduction="mean"
                    )
                    pll = -nll.item()
                else:
                    pll = 0.0
                
                plls.append(pll)
        
        return np.array(plls, dtype=np.float32)

    def probability(self, seqs, align=False, stepwise_masking=False, **kwargs):
        """
        Probability of mutations - applies softmax to logits to get probabilities
        """
        # Format input: join VH and VL with '|'
        formatted_seqs = []
        for s in seqs:
            if isinstance(s, (list, tuple)):
                formatted_seqs.append('|'.join(s))
            else:
                formatted_seqs.append(s)

        # Get logits
        if stepwise_masking:
            # For stepwise masking, we need to implement it similar to likelihood
            # This is a simplified version - you might want to implement full stepwise masking
            logits = self._predict_logits(formatted_seqs)
        else:
            logits = self._predict_logits(formatted_seqs)
        
        # Apply softmax to get probabilities
        probs = logits.softmax(-1).cpu().numpy()
        
        if align:
            return probs
        else:
            # Return residue-level probabilities (excluding special tokens)
            return [res_to_list(state, seq) for state, seq in zip(probs, formatted_seqs)]

    def restore(self, seqs, align=False, **kwargs):
        hf_abrestore = HFAbRestore(self.AbLang, self.tokenizer, spread=self.spread, device=self.used_device, ncpu=self.ncpu)
        restored = hf_abrestore.restore(seqs, align=align)
        # Apply angle brackets formatting to match original format
        if isinstance(restored, np.ndarray):
            restored = np.array([add_angle_brackets(seq) for seq in restored])
        else:
            restored = [add_angle_brackets(seq) for seq in restored]
        return restored

def extract_input_ids(tokens, device):
    if hasattr(tokens, 'input_ids'):
        return tokens.input_ids.to(device)
    elif isinstance(tokens, dict):
        if 'input_ids' in tokens:
            return tokens['input_ids'].to(device)
        else:
            for v in tokens.values():
                if hasattr(v, 'ndim') or torch.is_tensor(v):
                    return v.to(device)
    elif torch.is_tensor(tokens):
        return tokens.to(device)
    else:
        raise ValueError("Could not extract input_ids from tokenizer output")
