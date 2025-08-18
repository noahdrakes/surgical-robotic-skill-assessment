import torch
from torch.utils.data import Dataset
import pandas as pd

class MetricsMLPDataset(Dataset):
    """
    CSV layout (per row):
        [trial_id, f1, f2, ..., fN, label]

    - First column: trial identifier (stored for reference)
    - Middle columns: features
    - Last column: string label in {novice, intermediate, expert} (case-insensitive)

    This dataset is for CLASSIFICATION: it maps labels -> class indices.
    """
    def __init__(
        self,
        csv_path: str,
        label_order=("NOVICE", "INTERMEDIATE", "EXPERT"),
        x_dtype: torch.dtype = torch.float32,
    ):
        # Read CSV
        df = pd.read_csv(csv_path)
        if df.shape[1] < 3:
            raise ValueError("CSV must have at least 3 columns: trial, features..., label")

        # Keep trial IDs (optional, useful for analysis)
        self.trial_ids = df.iloc[:, 0].astype(str).tolist()

        # Features are all middle columns
        feature_df = df.iloc[:, 1:-1]
        self.X = torch.tensor(feature_df.values, dtype=x_dtype)

        # Labels: last column -> class indices according to label_order
        raw_labels = (df.iloc[:, -1].astype(str))

        # mapping labels to ints
        self.label_mapping = {label_order[0]: 0, label_order[1]: 1, label_order[2]: 2}
        
        y_idx = [self.label_mapping[val] for val in raw_labels]
        self.y = torch.tensor(y_idx, dtype=torch.long)

        # Convenience attributes
        self.n_features = self.X.shape[1]
        self.n_classes = len(self.label_mapping)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# from torch.utils.data import DataLoader

# ds = MetricsMLPDataset("../metrics/results.csv")
# loader = DataLoader(ds, batch_size=256, shuffle=True)

# for xb, yb in loader:
#     # train step
#     print("FEATS: ", xb)
#     print("LABEL: ", yb)
#     pass
