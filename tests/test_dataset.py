import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from src.data.dataset import ValidationDataset


@pytest.fixture
def setup_data(tmp_path):
    """
    Set up a temporary dataset with mocked images and a parquet file.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create mock image files
    num_samples = 10
    image_paths = []
    for i in range(num_samples):
        img_path = data_dir / f"image_{i}.jpg"
        image = Image.new("RGB", (100, 100), color=(i * 10, i * 10, i * 10))
        image.save(img_path)
        image_paths.append(f"image_{i}.jpg")

    # Create mock parquet data
    df = pd.DataFrame({
        "image": image_paths,
        "text": [f"Sample text {i}" for i in range(num_samples)],
        "label": np.random.randint(0, 2, size=num_samples),
    })
    df.to_parquet(data_dir / "dataset.parquet")

    return data_dir


def test_dataset_initialization(setup_data):
    """
    Test the initialization of ValidationDataset and split sizes.
    """
    data_dir = setup_data

    # Initialize dataset for each split
    train_dataset = ValidationDataset(data_dir=str(data_dir), split="train")
    val_dataset = ValidationDataset(data_dir=str(data_dir), split="val")
    test_dataset = ValidationDataset(data_dir=str(data_dir), split="test")

    # Assert splits match expected integer sizes
    total_samples = 10
    train_size = int(total_samples * 0.7)
    val_size = int(total_samples * 0.15)
    test_size = total_samples - train_size - val_size  # Remaining samples

    assert len(train_dataset) == train_size
    assert len(val_dataset) == val_size
    assert len(test_dataset) == test_size


def test_getitem(setup_data):
    """
    Test the __getitem__ method.
    """
    data_dir = setup_data

    # Initialize dataset
    dataset = ValidationDataset(data_dir=str(data_dir), split="train")

    # Retrieve a sample
    sample = dataset[0]

    # Assert the sample contains expected keys
    assert "image" in sample
    assert "text" in sample
    assert "label" in sample

    # Check types and dimensions
    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].shape[0] == 3  # RGB channels
    assert isinstance(sample["text"], str)
    assert isinstance(sample["label"], torch.Tensor)
    assert sample["label"].dtype == torch.float32


def test_custom_transform(setup_data):
    """
    Test dataset with a custom transformation pipeline.
    """
    data_dir = setup_data

    # Define custom transformation
    custom_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Initialize dataset with custom transform
    dataset = ValidationDataset(data_dir=str(data_dir), split="train", transform=custom_transform)

    # Retrieve a sample
    sample = dataset[0]
    image = sample["image"]

    # Assert the image size matches the transformation
    assert image.shape == torch.Size([3, 128, 128])


def test_missing_image_raises_error(setup_data):
    """
    Test that a missing image file raises a FileNotFoundError.
    """
    data_dir = setup_data

    # Initialize dataset
    dataset = ValidationDataset(data_dir=str(data_dir), split="train")

    # Get the path of the first image and remove it
    first_image_path = data_dir / dataset.df.iloc[0]["image"]
    if first_image_path.exists():
        first_image_path.unlink()

    # Reinitialize dataset to ensure fresh state
    dataset = ValidationDataset(data_dir=str(data_dir), split="train")

    # Access the first item, which should now raise a FileNotFoundError
    with pytest.raises(FileNotFoundError):
        _ = dataset[0]
