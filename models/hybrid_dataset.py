"""Two-input dataset + per-node image scaler for the CNN arm."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from models.dataset import WindowConfig
from models.surface_cnn import GRID_SHAPE


class SurfaceScaler:
    """Per-node standardization of the shape grid, fit on training rows
    only - the image analogue of models.dataset.FeatureScaler."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, df: pd.DataFrame):
        self.mean = df.mean()
        self.std = df.std().replace(0, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return (df - self.mean) / self.std


class SurfaceRVDataset(Dataset):
    """Windows of (baseline features, surface images) -> log target.

    Same valid-index logic as models.dataset.RVDataset; __getitem__
    additionally reshapes each day's 228 shape columns into a
    (seq_len, 1, 12, 19) image stack. Inputs must already share an index
    (build_cnn_inputs guarantees it); asserted, not silently re-aligned.
    """

    def __init__(self, features: pd.DataFrame, surface: pd.DataFrame,
                 target: pd.Series, config: WindowConfig):
        assert features.index.equals(surface.index), "align inputs upstream"
        common = features.index.intersection(target.index)
        self.features = features.loc[common]
        self.surface = surface.loc[common]
        self.target = target.loc[common]
        self.config = config
        self.valid_indices = []
        for i in range(config.seq_len - 1, len(self.features)):
            date = self.features.index[i]
            if date not in self.target.index:
                continue
            y = self.target.loc[date]
            if pd.notna(y) and np.isfinite(y) and y > 0:
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        sl = slice(i - self.config.seq_len + 1, i + 1)
        x_base = torch.tensor(self.features.iloc[sl].values,
                              dtype=torch.float32)
        imgs = self.surface.iloc[sl].values.reshape(-1, 1, *GRID_SHAPE)
        x_surf = torch.tensor(imgs, dtype=torch.float32)
        y = np.log(float(self.target.iloc[i]))
        return (x_base, x_surf), torch.tensor([y], dtype=torch.float32)

    def dates(self):
        return [self.features.index[i] for i in self.valid_indices]
