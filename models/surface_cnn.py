"""Convolutional encoder for daily vol-surface images + transformer pairing.

Block design follows Kelly, Kuznetsov, Malamud & Xu (their Fig. 3):
3x3 conv -> ReLU -> 2x2 max-pool -> batch norm, filter counts growing
through the stack, global average pooling on top. Instead of their single
dense forecast node, the head emits a d_embed-dimensional embedding that
is concatenated with the baseline features and fed to the UNCHANGED
production RVTransformer.
"""
import torch
import torch.nn as nn

from models.transformer import RVTransformer

GRID_SHAPE = (12, 19)          # maturities x deltas, matches fit_surface.py
MAX_POOLED_BLOCKS = 3          # 12x19 -> 6x9 -> 3x4 -> 1x2; a 4th pool dies


class SurfaceCNN(nn.Module):
    """(N, 1, 12, 19) shape images -> (N, d_embed) embeddings."""

    def __init__(self, channels=(16, 32, 64), d_embed=16):
        super().__init__()
        layers, prev = [], 1
        for i, ch in enumerate(channels):
            layers.append(nn.Conv2d(prev, ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            if i < MAX_POOLED_BLOCKS:      # deeper blocks convolve, no pool
                layers.append(nn.MaxPool2d(2))
            layers.append(nn.BatchNorm2d(ch))
            prev = ch
        self.blocks = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(prev, d_embed)
        self.d_embed = d_embed
        for m in self.modules():           # paper: Xavier initialization
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(x)                 # (N, C_last, 1, 2) at depth 3
        h = self.gap(h).flatten(1)         # (N, C_last)
        return self.head(h)                # (N, d_embed)


class SurfaceRVTransformer(nn.Module):
    """CNN eye + transformer memory, trained end-to-end.

    Input is a tuple (x_base, x_surf):
      x_base: (B, L, n_base)      scaled baseline features + surf_level
      x_surf: (B, L, 1, 12, 19)   standardized shape images

    One encoder, shared across all B*L days; per-day embeddings are
    concatenated with x_base and handed to the production transformer,
    whose input projection resizes itself via n_features.
    """

    def __init__(self, n_base, channels=(16, 32, 64), d_embed=16,
                 **transformer_kwargs):
        super().__init__()
        self.encoder = SurfaceCNN(channels=channels, d_embed=d_embed)
        self.transformer = RVTransformer(
            n_features=n_base + d_embed, **transformer_kwargs)

    def forward(self, x):
        x_base, x_surf = x
        B, L = x_surf.shape[:2]
        emb = self.encoder(x_surf.reshape(B * L, *x_surf.shape[2:]))
        emb = emb.reshape(B, L, -1)
        return self.transformer(torch.cat([x_base, emb], dim=-1))
