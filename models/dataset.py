from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

@dataclass
class WindowConfig: 
    seq_len: int = 60
    horizon: int = 21 #informational

class RVDataset(Dataset):
    def __init__(
            self, 
            features: pd.DataFrame,
            target: pd.Series,
            config: WindowConfig, 
    ):
        
        common = features.index.intersection(target.index)
        self.features = features.loc[common]
        self.target = target.loc[common]
        self.config = config

        self.valid_indices = []
        for i in range(config.seq_len-1, len(self.features)):
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
        x = self.features.iloc[i - self .config.seq_len + 1 : i + 1].values
        y_raw = float(self.target.iloc[i])
        if not np.isfinite(y_raw) or y_raw <= 0:
            raise ValueError(f"Non-positive/NaN target at {self.features.index[i]}: {y_raw}")
        y = np.log(y_raw)
        return torch.tensor(x, dtype=torch.float32), torch.tensor([y], dtype=torch.float32)
    def dates(self):
        return [self.features.index[i] for i in self.valid_indices]
    
class FeatureScaler: 
    def __init__(self):
        self.mean = None
        self.std = None
    def fit(self , df: pd.DataFrame):
        self.mean = df.mean()
        self.std = df.std().replace(0, 1.0)
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return (df - self.mean) / self.std