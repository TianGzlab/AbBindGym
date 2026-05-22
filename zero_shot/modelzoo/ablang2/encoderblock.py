import torch
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
