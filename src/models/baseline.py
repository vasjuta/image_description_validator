import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


class BaselineValidator(nn.Module):
    """Baseline model for image-text validation using CLIP."""

    def __init__(
            self,
            clip_model: str = "openai/clip-vit-base-patch32",
            freeze_clip: bool = True
    ):
        super().__init__()

        # Load CLIP model and processor
        self.clip = CLIPModel.from_pretrained(clip_model)
        self.processor = CLIPProcessor.from_pretrained(clip_model)

        # Freeze CLIP parameters if specified
        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False

        # Classification head
        clip_dim = self.clip.config.projection_dim
        self.classifier = nn.Sequential(
            nn.Linear(1, 32),  # Adjust dimensions to fit similarity input
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, images, texts):
        """
        Forward pass of the model.

        Args:
            images: Tensor of images (B, C, H, W)
            texts: List of text descriptions

        Returns:
            Tensor of probabilities (B, 1)
        """
        # Get CLIP features
        inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,  # CLIP's max length
            do_rescale=False
        ).to(images.device)

        clip_outputs = self.clip(**inputs)

        # Extract diagonal elements (similarities between corresponding pairs)
        similarity = torch.diag(clip_outputs.logits_per_image)  # (batch_size,)

        # Reshape to (batch_size, 1)
        similarity = similarity.unsqueeze(1)  # Ensure correct shape (batch_size, 1)

        # Classification
        return self.classifier(similarity)
