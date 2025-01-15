import argparse
from pathlib import Path

from src.utils.eda import DatasetAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description="Run exploratory data analysis")
    parser.add_argument("data_dir", type=str, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="outputs/eda",
                        help="Directory to save EDA outputs")
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis
    analyzer = DatasetAnalyzer(args.data_dir)

    # Print and save summary
    analyzer.print_summary(output_dir / "summary.txt")

    # Generate and save visualizations
    analyzer.generate_visualizations(output_dir)
    print(f"\nEDA outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()