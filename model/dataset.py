import torch
from torch.utils.data import Dataset
import pandas as pd

class MetricsMLPDataset(Dataset):
    DEFAULT_FEATURES = [
        "completion_time",
        "PSM1_average_speed_magnitude",
        "PSM2_average_speed_magnitude",
        "PSM1_average_acceleration_magnitude",
        "PSM2_average_acceleration_magnitude",
        "PSM1_average_jerk_magnitude",
        "PSM2_average_jerk_magnitude",
        "ATIForceSensor_average_force_magnitude",
        "PSM1_total_path_length",
        "PSM2_total_path_length",
        "PSM1_average_angular_speed_magnitude",
        "PSM2_average_angular_speed_magnitude",
        "PSM1_PSM2_speed_correlation",
        "PSM1_PSM2_speed_cross",
        "PSM1_PSM2_acceleration_cross",
        "PSM1_PSM2_jerk_cross",
        "PSM1_PSM2_acceleration_dispertion",
        "PSM1_PSM2_jerk_dispertion",
        "PSM1_forcen_magnitude",
        "PSM2_forcen_magnitude",
    ]
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
        norm_mean: torch.Tensor | None = None,
        norm_std: torch.Tensor | None = None,
        normalize: bool = True,
        features: list[str] | None = None,
    ):
        # Read CSV
        df = pd.read_csv(csv_path)
        if df.shape[1] < 3:
            raise ValueError("CSV must have at least 3 columns: trial, features..., label")

        # Keep trial IDs (optional, useful for analysis)
        self.trial_ids = df.iloc[:, 0].astype(str).tolist()
        self.subject_ids = [tid.split("_")[0] for tid in self.trial_ids]
        self.trial_nums = [tid.split("_")[1] if "_" in tid else None for tid in self.trial_ids]

        # Exclude these columns from features
        exclude_cols = {"Subject_Trial", "LABEL", "subject_id", "trial_id"}
        if features is None:
            print("FEATURES IS NONE")
            # Use all columns except exclude_cols
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            feature_df = df[feature_cols]
        else:
            # Validate that all chosen features are present in df.columns and not in exclude_cols
            missing = [f for f in features if f not in df.columns]
            if missing:
                raise ValueError(f"Features not found in CSV columns: {missing}")
            # Remove any excluded columns from features
            filtered_features = [f for f in features if f not in exclude_cols]
            feature_df = df[filtered_features]
        self.X = torch.tensor(feature_df.values, dtype=x_dtype)

        # Compute mean and std along dataset (dim=0: column-wise), or use provided stats
        if norm_mean is None or norm_std is None:
            self.feature_mean = self.X.mean(dim=0)
            self.feature_std = self.X.std(dim=0)
        else:
            self.feature_mean = norm_mean
            self.feature_std = norm_std

        # Clamp std for numerical safety
        self._std_safe = torch.clamp(self.feature_std, min=1e-12)

        self.normalize = normalize

        # Labels: use "LABEL" column, uppercase for consistency
        if "LABEL" not in df.columns:
            raise ValueError('CSV must contain a "LABEL" column.')
        raw_labels = df["LABEL"].astype(str).str.upper()

        # mapping labels to ints
        self.label_mapping = {label_order[0]: 0, label_order[1]: 1, label_order[2]: 2}
        
        y_idx = [self.label_mapping[val] for val in raw_labels]
        self.y = torch.tensor(y_idx, dtype=torch.long)

        # Convenience attributes
        self.n_features = self.X.shape[1]
        self.n_classes = len(self.label_mapping)
        self.n_samples = self.X.shape[0]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.normalize:
            return (self.X[idx] - self.feature_mean) / self._std_safe, self.y[idx]
        else:
            return self.X[idx], self.y[idx]
    

# Example usage:
# train_ds = MetricsMLPDataset("train.csv", normalize=True)
# val_ds = MetricsMLPDataset("val.csv", normalize=True, norm_mean=train_ds.feature_mean, norm_std=train_ds.feature_std)
