import argparse
import sys
from pathlib import Path
from PIL import Image, UnidentifiedImageError

from src.inference.api import ImageTextValidator
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Image-Text Validation Client")
    parser.add_argument("model_path", type=str, help="Path to saved model")
    parser.add_argument("image_path", type=str, help="Path to image file")
    parser.add_argument("text", type=str, help="Text description to validate")
    return parser.parse_args()


def validate_inputs(model_path: str, image_path: str, text: str) -> None:
    """Validate all inputs before processing."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")

    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")

    if not text or not text.strip():
        raise ValueError("Text description cannot be empty")

    # Try opening image to catch format issues early
    try:
        Image.open(image_path)
    except UnidentifiedImageError:
        raise ValueError(f"Could not open image at {image_path}. Make sure it's a valid image file.")


def main():
    try:
        args = parse_args()
        logger = setup_logger("inference")

        # Validate inputs
        validate_inputs(args.model_path, args.image_path, args.text)

        logger.info(f"Loading model from {args.model_path}")
        validator = ImageTextValidator(args.model_path)

        result = validator.validate(args.image_path, args.text)

        print("\nValidation Results:")
        print("-" * 20)
        print(f"Image: {args.image_path}")
        print(f"Text: {args.text}")
        print(f"Result: {result['result']}")
        print(f"Probability: {result['probability']:.4f}")

        return result  # Return result for programmatic usage

    except FileNotFoundError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()