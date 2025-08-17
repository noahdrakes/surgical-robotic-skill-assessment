import torch
from torch.utils.data import Dataset

class MetricsMLPDataset(Dataset):
    def __init__(self, df, feature_cols, label_col):
        self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df[label_col].values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]