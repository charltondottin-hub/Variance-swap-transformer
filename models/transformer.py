"""
Encoder only transformer for realized variance forecasting"""

import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, d_model:int, max_len:int =200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]

class EncoderBlock(nn.Module):
    """Single transformer encoder block with multi-head attention and feedforward layers"""
    def __init__(self, d_model:int, n_heads:int, d_ff:int, dropout:float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, 
                                          batch_first = True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),)         
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.Tensor, return_attn: bool = False):
        h = self.norm1(x)
        attn_out, attn_weights = self.attn(h, h, h, need_weights=return_attn, 
                                           average_attn_weights=False)
        x = x+ self.dropout(attn_out)
        h = self.norm2(x)
        x = x + self.dropout(self.ffn(h))
        return (x, attn_weights) if return_attn else x
    
class RVTransformer(nn.Module):
    """Encoder-only transformer that maps a sequence of feature vectors to a single 
    scalar prediction of log forward realized variance."""
    def __init__(self, 
                 n_features:int, 
                 d_model:int=64, 
                 n_heads:int=4, 
                 n_layers:int=2,
                 d_ff:int=128,
                 dropout:float=0.1,
                 max_len:int=200,):
     
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
    
    def forward(self, x:torch.Tensor, return_attn: bool = False):
        h = self.input_proj(x)
        h = self.pos_enc(h)
        attn_maps = []
        for block in self.blocks:
            if return_attn:
                h, attn = block(h, return_attn=True)
                attn_maps.append(attn)
            else:
                h = block(h)

        h = self.norm(h)
        last = h[:, -1, :]
        out = self.head(last)
        if return_attn:
            return out, attn_maps
        return out
