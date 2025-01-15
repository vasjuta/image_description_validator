import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import ValidationDataset
from src.models.baseline import BaselineValidator
from src.training.trainer import Trainer
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train image-text validator")
    parser.add_argument("data_dir", type=str, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save model outputs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # Setup logging
    logger = setup_logger(
        "training",
        log_file=log_dir / "training.log"
    )
    logger.info(f"Arguments: {args}")

    # Create datasets and dataloaders
    logger.info("Creating datasets...")
    train_dataset = ValidationDataset(args.data_dir, split="train")
    val_dataset = ValidationDataset(args.data_dir, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Initialize model and trainer
    logger.info("Initializing model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BaselineValidator().to(device)

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        learning_rate=args.learning_rate,
        device=device
    )

    # Train
    logger.info("Starting training...")
    trainer.train(
        num_epochs=args.num_epochs,
        checkpoint_dir=model_dir,
        patience=args.patience
    )

    logger.info("Training completed!")


if __name__ == "__main__":
    main()