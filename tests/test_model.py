import pytest
import torch
from PIL import Image
import numpy as np
from src.models.baseline import BaselineValidator
from src.utils.preprocessing import get_default_transform


@pytest.fixture
def model():
    return BaselineValidator()


@pytest.fixture
def transform():
    return get_default_transform()


@pytest.fixture
def sample_batch(transform):
    batch_size = 2

    # Create sample PIL images (random noise, but properly processed)
    images = []
    for _ in range(batch_size):
        # Create random RGB image array (0-255 range)
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        pil_image = Image.fromarray(img_array)
        # Apply the same transform pipeline as in the dataset
        processed_image = transform(pil_image)
        images.append(processed_image)

    # Stack images into a batch
    image_batch = torch.stack(images)

    return {
        "image": image_batch,
        "text": ["A dog on a beach", "A cat in a box"]
    }


def test_model_output_shape(model, sample_batch):
    with torch.no_grad():  # Avoid unnecessary gradient computation during testing
        outputs = model(sample_batch["image"], sample_batch["text"])
    assert outputs.shape == (2, 1)
    assert torch.all((outputs >= 0) & (outputs <= 1))


def test_model_gradient_flow(model, sample_batch):
    model.train()
    outputs = model(sample_batch["image"], sample_batch["text"])
    loss = outputs.mean()
    loss.backward()

    # Check if gradients are flowing to trainable parameters
    grad_norm = 0
    for param in model.classifier.parameters():
        if param.grad is not None:
            grad_norm += param.grad.norm().item()

    assert grad_norm > 0, "No gradients in classifier parameters"


def test_model_freeze_clip(transform):
    # Test with frozen CLIP
    model_frozen = BaselineValidator(freeze_clip=True)
    for param in model_frozen.clip.parameters():
        assert not param.requires_grad

    # Test with unfrozen CLIP
    model_unfrozen = BaselineValidator(freeze_clip=False)
    has_trainable = False
    for param in model_unfrozen.clip.parameters():
        if param.requires_grad:
            has_trainable = True
            break
    assert has_trainable
