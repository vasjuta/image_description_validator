from torchvision import transforms


def get_default_transform():
    """Get default transformation pipeline for images."""
    return transforms.Compose([
        transforms.Resize((256, 256)),  # Ensure standardized size
        transforms.ToTensor(),          # Convert to PyTorch tensor
    ])