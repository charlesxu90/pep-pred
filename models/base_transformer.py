from collections import OrderedDict

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
import logging

logger = logging.getLogger(__name__)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, dropout: float = 0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask  # support both BERT and GPT depending on the attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, 
                 grad_ckpt: bool = False, dropout: float = 0., emb_dropout: float = 0.):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_ckpt = grad_ckpt
        self.dropout = nn.Dropout(emb_dropout)
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, dropout=dropout) for _ in range(layers)])

    def ckpt_fwd(self, layer, input, segments=2):
        """checkpoint forward"""
        if not input.requires_grad:  # Enable computing gradients for the input
            input = input.detach()
            input.requires_grad = True
        return checkpoint_sequential(layer, segments, input)

    def forward(self, x: torch.Tensor):
        x = self.dropout(x)
        if self.grad_ckpt:  # Reduce memory consumption by checkpointing the forward pass
            return self.ckpt_fwd(self.resblocks, x, self.layers)
        return self.resblocks(x)


class DecoderAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, source_attn_mask: torch.Tensor = None, dropout: float = 0.):
        super().__init__()

        self.attn_mask = attn_mask  # support both BERT and GPT depending on the attn_mask
        self.source_attn_mask = source_attn_mask

        self.ln_1 = LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_2 = LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_3 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.self_attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)[0]
    
    def cross_attention(self, x: torch.Tensor, hidden_states=None):
        self.source_attn_mask = self.source_attn_mask.to(dtype=x.dtype, device=x.device) if self.source_attn_mask is not None else None
        return self.cross_attn(x, hidden_states, hidden_states, need_weights=True, attn_mask=self.source_attn_mask)[0]

    def forward(self, x: torch.Tensor, hidden_states=None):
        x = x + self.attention(self.ln_1(x))
        x = x + self.cross_attention(self.ln_2(x), hidden_states)
        x = x + self.mlp(self.ln_3(x))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, width: int, n_layers: int, n_heads: int, 
                 attn_mask: torch.Tensor = None, source_attn_mask: torch.Tensor = None,
                 dropout: float = 0., emb_dropout: float = 0.):
        super().__init__()
        self.width = width
        self.n_layers = n_layers
        self.dropout = nn.Dropout(emb_dropout)
        self.resblocks = nn.ModuleList([DecoderAttentionBlock(width, n_heads, attn_mask, source_attn_mask, dropout=dropout) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, hidden_states=None):
        x = self.dropout(x)
        for block in self.resblocks:
            x = block(x, hidden_states)
        return x
