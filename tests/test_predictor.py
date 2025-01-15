import pytest
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from src.inference.predictor import ImageTextPredictor
from src.models.model import ImageTextModel


@pytest.fixture
def model_path(tmp_path):
    # Create and save dummy model
    model = ImageTextModel()
    path = tmp_path / "model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': None
    }, path)
    return path


@pytest.fixture
def dummy_image():
    return Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))


def test_predictor_initialization(model_path):
    predictor = ImageTextPredictor(model_path)
    assert predictor.model is not None
    assert predictor.tokenizer is not None


def test_prediction(model_path, dummy_image):
    predictor = ImageTextPredictor(model_path)
    result = predictor.predict(dummy_image, "A test description")

    assert isinstance(result, dict)
    assert "probability" in result
    assert "prediction" in result
    assert "label_text" in result
    assert 0 <= result["probability"] <= 1
    assert result["prediction"] in [0, 1]
    assert result["label_text"] in ["Match", "No match"]