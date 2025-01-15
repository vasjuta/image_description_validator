import pytest
import torch
from src.models.baseline import BaselineValidator


@pytest.fixture
def model():
    return BaselineValidator()


@pytest.fixture
def sample_batch():
    batch_size = 2
    return {
        "image": torch.randn(batch_size, 3, 256, 256),
        "text": ["A dog on a beach", "A cat in a box"]
    }


def test_model_output_shape(model, sample_batch):
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


def test_model_freeze_clip(sample_batch):
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