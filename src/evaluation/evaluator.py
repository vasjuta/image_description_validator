from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score
)

from ..utils.logger import setup_logger

logger = setup_logger("evaluator")


class Evaluator:
    """Evaluator for image-text validation models."""

    def __init__(
            self,
            model: torch.nn.Module,
            data_loader: DataLoader,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize evaluator.

        Args:
            model: Model to evaluate
            data_loader: DataLoader for evaluation data
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation.

        Returns:
            Dict containing evaluation metrics
        """
        self.model.eval()
        predictions = []
        labels = []
        raw_probabilities = []

        logger.info("Starting evaluation...")
        for batch in tqdm(self.data_loader):
            images = batch["image"].to(self.device)
            texts = batch["text"]
            batch_labels = batch["label"]

            outputs = self.model(images, texts)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            predictions.extend(preds)
            labels.extend(batch_labels.numpy())
            raw_probabilities.extend(probs)

        predictions = np.array(predictions)
        labels = np.array(labels)
        raw_probabilities = np.array(raw_probabilities)

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )

        conf_matrix = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = conf_matrix.ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        auc_roc = roc_auc_score(labels, raw_probabilities)

        metrics = {
            "accuracy": (tp + tn) / (tp + tn + fp + fn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "auc_roc": auc_roc
        }

        logger.info("Evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        return metrics

    def analyze_errors(self, output_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        Analyze prediction errors in detail.

        Args:
            output_dir: Optional directory to save error analysis

        Returns:
            DataFrame containing error analysis
        """
        self.model.eval()
        error_cases = []

        logger.info("Analyzing errors...")
        with torch.no_grad():
            for batch in tqdm(self.data_loader):
                images = batch["image"].to(self.device)
                texts = batch["text"]
                labels = batch["label"]

                outputs = self.model(images, texts)
                probs = torch.sigmoid(outputs).cpu().numpy()
                predictions = (probs >= 0.5).astype(int)

                # Collect error cases
                for text, label, pred, prob in zip(
                        texts, labels, predictions, probs
                ):
                    if label != pred:
                        error_cases.append({
                            "text": text,
                            "true_label": label.item(),
                            "predicted": pred.item(),
                            "confidence": prob.item()
                        })

        error_df = pd.DataFrame(error_cases)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            error_df.to_csv(output_dir / "error_analysis.csv", index=False)
            logger.info(f"Error analysis saved to {output_dir}/error_analysis.csv")

        return error_df