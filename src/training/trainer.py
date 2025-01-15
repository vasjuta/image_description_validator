from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Trainer:
    """Trainer for the baseline model."""

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            learning_rate: float = 1e-4,
            weight_decay: float = 0.01,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Calculate class weights for loss
        n_samples = len(train_loader.dataset)
        n_positives = sum(train_loader.dataset.df["label"])
        pos_weight = (n_samples - n_positives) / n_positives
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        predictions, labels = [], []

        for batch in tqdm(self.train_loader, desc="Training"):
            # Move data to device
            images = batch["image"].to(self.device)
            texts = batch["text"]
            batch_labels = batch["label"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, texts)
            loss = self.criterion(outputs, batch_labels.view(-1, 1))

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Store predictions and labels
            total_loss += loss.item()
            predictions.extend((outputs >= 0).cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )

        return {
            "loss": total_loss / len(self.train_loader),
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        predictions, labels = [], []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch["image"].to(self.device)
                texts = batch["text"]
                batch_labels = batch["label"].to(self.device)

                outputs = self.model(images, texts)
                loss = self.criterion(outputs, batch_labels.view(-1, 1))

                total_loss += loss.item()
                predictions.extend((outputs >= 0).cpu().numpy())
                labels.extend(batch_labels.cpu().numpy())

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )

        return {
            "val_loss": total_loss / len(self.val_loader),
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1
        }

    def train(
            self,
            num_epochs: int,
            checkpoint_dir: Optional[Path] = None,
            patience: int = 3
    ):
        """Full training loop with early stopping."""
        import time
        start_time = time.time()
        best_f1 = 0
        patience_counter = 0

        for epoch in range(num_epochs):
            epoch_start = time.time()
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train and validate
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            # Calculate epoch time
            epoch_time = time.time() - epoch_start

            # Print metrics
            metrics = {**train_metrics, **val_metrics}
            print("\n".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
            print(f"Epoch time: {epoch_time:.2f} seconds")

            # Early stopping
            if val_metrics["val_f1"] > best_f1:
                best_f1 = val_metrics["val_f1"]
                patience_counter = 0

                # Save best model
                if checkpoint_dir:
                    self.save_checkpoint(checkpoint_dir / "best_model.pt")
                    print(f"Saved best model with validation F1: {best_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping after {epoch + 1} epochs")
                    break

        total_time = time.time() - start_time
        print(f"\nTotal training time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
        print(f"Best validation F1: {best_f1:.4f}")

    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)