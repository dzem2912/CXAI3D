import os
import torch
import random
import warnings

import pandas as pd
import numpy as np

from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

class PointcloudDataset(Dataset):
    def __init__(self, root: str, num_points: int = 4096, fraction: float = 1.0):
        self.root = root
        self.files = [os.path.join(self.root, f) for f in os.listdir(self.root) if f.endswith('.csv')]

        # Specify if you want to train the model on a fraction of a dataset instead of the whole dataset
        if fraction < 1.0:
            num_samples = int(len(self.files) * fraction)
            print(f"Fraction: {fraction:.2f}; Dataset downsampled from {len(self.files)} to {num_samples}")
            self.files = random.sample(self.files, k=num_samples)

        self.num_points = num_points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = self.files[index]
        sample: pd.DataFrame = pd.read_csv(file_path, sep=',')
        original_size: int = len(sample)

        points: np.ndarray = sample[['x', 'y', 'z']].to_numpy()

        if not all(col in sample.columns for col in ['b', 'g', 'b']):
            colors: np.ndarray = np.zeros_like(points)
        else:
            colors: np.ndarray = sample[['r', 'g', 'b']].to_numpy()

        if 'Intensity' not in sample.columns:
            intensity: np.ndarray = np.zeros((original_size, 1))
        else:
            intensity: np.ndarray = sample['Intensity'].to_numpy().reshape(-1, 1)

        if 'nr' not in sample.columns:
            num_returns: np.ndarray = np.zeros((original_size, 1))
        else:
            num_returns: np.ndarray = (sample['nr'].to_numpy().reshape(-1, 1))

        features: np.ndarray = np.hstack((points, colors, intensity, num_returns))
        targets: np.ndarray = sample['sem_class'].to_numpy()

        selected_indices = np.random.choice(len(targets), size=self.num_points, replace=False)

        features = features[selected_indices, :]
        targets = targets[selected_indices]

        features: torch.Tensor = torch.tensor(features, dtype=torch.float64)
        targets: torch.Tensor = torch.tensor(targets, dtype=torch.int64)

        return features, targets
