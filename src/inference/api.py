from pathlib import Path
from typing import Union, Dict
import base64
from io import BytesIO

import torch
from PIL import Image
import numpy as np

from ..models.baseline import BaselineValidator
from ..utils.preprocessing import get_default_transform


class ImageTextValidator:
    """User-friendly API for image-text validation."""

    def __init__(self, model_path: Union[str, Path]):
        """
        Initialize the validator.

        Args:
            model_path: Path to the trained model checkpoint
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize model
        self.model = BaselineValidator().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Default transformations
        # todo: take from config
        self.transform = get_default_transform()

    def validate(
            self,
            image: Union[str, Path, Image.Image, bytes],
            text: str
    ) -> Dict[str, Union[float, str]]:
        """
        Validate if text description matches the image.

        Args:
            image: Image input. Can be:
                  - Path to image file
                  - PIL Image object
                  - Base64 encoded image string
                  - Bytes of image
            text: Text description to validate

        Returns:
            Dictionary containing:
            - probability: Match probability (0-1)
            - prediction: Binary prediction (0 or 1)
            - label_text: Human-readable result
        """
        # Input validation
        if not text or not text.strip():
            raise ValueError("Text description cannot be empty")

        # Process image
        if isinstance(image, (str, Path)):
            if str(image).startswith('data:image'):  # Base64 string
                image = Image.open(BytesIO(base64.b64decode(
                    image.split(',')[1])))
            else:  # File path
                image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        elif not isinstance(image, Image.Image):
            raise ValueError(
                "Image must be a file path, PIL Image, base64 string, or bytes"
            )

        # Ensure image is RGB
        image = image.convert('RGB')

        # Get model prediction
        # Get model prediction
        with torch.no_grad():
            # Process inputs
            processed = self.model.processor(
                images=image,
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
                do_rescale=False
            ).to(self.device)

            # Pass to model with expected arguments
            outputs = self.model(processed['pixel_values'], [text])
            probability = torch.sigmoid(outputs)[0].item()

        # Format results
        prediction = int(probability >= 0.5)
        label_text = "Match" if prediction == 1 else "No match"
        confidence = max(probability, 1 - probability) * 100

        return {
            "probability": probability,
            "prediction": prediction,
            "label_text": label_text,
            "confidence": f"{confidence:.1f}%",
            "result": f"{label_text} (confidence: {confidence:.1f}%)"
        }

    def __call__(self, image: Union[str, Path, Image.Image], text: str) -> Dict:
        """Convenience method to call validate()."""
        return self.validate(image, text)