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

        # Get CLIP dimensions
        clip_dim = self.clip.config.projection_dim

        # New classifier that uses both similarity and raw features
        self.classifier = nn.Sequential(
            # Concatenated: similarity (1) + image features (clip_dim) + text features (clip_dim)
            nn.Linear(2 * clip_dim + 1, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
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
            max_length=77,
            do_rescale=False
        ).to(images.device)

        clip_outputs = self.clip(**inputs)

        # Get image and text features
        image_features = clip_outputs.image_embeds
        text_features = clip_outputs.text_embeds

        # Get similarity score
        similarity = torch.diag(clip_outputs.logits_per_image).unsqueeze(1)

        # print(f"Similarity range: {similarity.min().item():.3f} to {similarity.max().item():.3f}")

        # Concatenate all features
        combined_features = torch.cat([similarity, image_features, text_features], dim=1)

        # Classification
        return self.classifier(combined_features)