import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import ValidationDataset
from src.models.baseline import BaselineValidator
from src.evaluation.evaluator import Evaluator
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate image-text validator")
    parser.add_argument("data_dir", type=str, help="Path to dataset directory")
    parser.add_argument("model_path", type=str, help="Path to saved model")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save evaluation outputs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)

    # Setup logging
    logger = setup_logger(
        "evaluation",
        log_file=eval_dir / "evaluation.log"
    )
    logger.info(f"Arguments: {args}")

    # Load test dataset explicitly
    logger.info("Loading test dataset...")
    dataset = ValidationDataset(args.data_dir, split="test")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BaselineValidator().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Run evaluation
    evaluator = Evaluator(model, dataloader, device)
    metrics = evaluator.evaluate()

    # Save metrics
    metrics_file = eval_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {metrics_file}")

    # Run error analysis
    logger.info("Running error analysis...")
    evaluator.analyze_errors(eval_dir)

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()