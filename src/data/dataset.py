from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

from src.utils.preprocessing import get_default_transform


class ValidationDataset(Dataset):
    """Dataset for image-text validation pairs."""

    def __init__(self, data_dir: str, split: str = "train", transform=None):
        """
        Initialize dataset.

        Args:
            data_dir: Root directory containing dataset.parquet and images
            split: Dataset split ("train", "val", or "test")
            transform: Transformations to apply to images (default: Resize and ToTensor)
        """
        self.data_dir = Path(data_dir)

        # Load and split data
        df = pd.read_parquet(self.data_dir / "dataset.parquet")

        # Create train/val/test splits (70/15/15)
        train_size = 0.7
        val_size = 0.15
        # Use fixed random seed for reproducibility
        rng = np.random.RandomState(42)

        # First split off the test set
        train_val_df = df.sample(frac=0.85, random_state=rng)
        test_df = df.drop(train_val_df.index)

        # Then split train/val from the remaining data
        train_df = train_val_df.sample(frac=train_size / (train_size + val_size),
                                       random_state=rng)
        val_df = train_val_df.drop(train_df.index)

        # Select the appropriate split
        if split == "train":
            self.df = train_df
        elif split == "val":
            self.df = val_df
        else:  # test
            self.df = test_df

        # Default transformations if none are provided
        self.transform = transform or get_default_transform()

        print(f"Loaded {len(self.df)} samples for {split} split")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Get a single image-text pair."""
        row = self.df.iloc[idx]

        # Convert relative path to absolute using data_dir
        img_path = self.data_dir / row["image"]
        if not img_path.exists():
            # Try without 'dataset' prefix if it exists in the path
            img_path = self.data_dir / row["image"].replace('dataset/', '')

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path} (original: {row['image']})")

        image = Image.open(img_path).convert("RGB")

        # Apply transformations
        image = self.transform(image)

        return {
            "image": image,
            "text": row["text"],
            "label": torch.tensor(row["label"], dtype=torch.float32)
        }
