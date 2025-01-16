import pytest
import torch
from PIL import Image
import numpy as np

from src.inference.api import ImageTextValidator
from src.models.baseline import BaselineValidator


@pytest.fixture
def model_path(tmp_path):
    # Create and save dummy model
    model = BaselineValidator()
    path = tmp_path / "model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': None
    }, path)
    return path


@pytest.fixture
def dummy_image():
    return Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))


def test_validator_initialization(model_path):
    validator = ImageTextValidator(model_path)
    assert validator.model is not None


def test_validation(model_path, dummy_image):
    validator = ImageTextValidator(model_path)
    result = validator.validate(dummy_image, "A test description")

    assert isinstance(result, dict)
    assert "probability" in result
    assert "prediction" in result
    assert "label_text" in result
    assert 0 <= result["probability"] <= 1
    assert result["prediction"] in [0, 1]
    assert result["label_text"] in ["Match", "No match"]