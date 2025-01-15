from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


class DatasetAnalyzer:
    """Analyzer for the image-text dataset statistics."""

    def __init__(self, data_dir: str):
        """
        Initialize analyzer.

        Args:
            data_dir: Root directory containing dataset.parquet and images folder
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.parquet_path = self.data_dir / "dataset.parquet"

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.parquet_path}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        # Read parquet file
        self.df = pd.read_parquet(self.parquet_path)

    def analyze_folder_structure(self) -> Dict:
        """Analyze the image folder structure."""
        folder_counts = {}
        for folder in sorted(self.images_dir.glob("[0-9]" * 5)):
            n_images = len(list(folder.glob("*.jpg")))
            folder_counts[folder.name] = n_images
        return folder_counts

    def analyze_image_sizes(self, sample_size: int = 1000) -> List[Tuple[int, int]]:
        """Analyze dimensions of a sample of images."""
        all_images = list(self.images_dir.rglob("*.jpg"))
        if len(all_images) > sample_size:
            sampled_images = np.random.choice(all_images, sample_size, replace=False)
        else:
            sampled_images = all_images

        sizes = []
        for img_path in tqdm(sampled_images, desc="Analyzing image sizes"):
            try:
                with Image.open(img_path) as img:
                    sizes.append(img.size)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        return sizes

    def analyze(self) -> Dict:
        """Run comprehensive analysis on the dataset."""
        stats = {}

        # Basic dataset stats
        stats['general'] = {
            'total_pairs': len(self.df),
            'total_folders': len(list(self.images_dir.glob("[0-9]" * 5))),
            'folder_counts': self.analyze_folder_structure()
        }

        # Label distribution if available
        if 'label' in self.df.columns:
            label_counts = self.df['label'].value_counts()
            stats['labels'] = {
                'distribution': label_counts.to_dict(),
                'positive_ratio': label_counts.get(1, 0) / len(self.df)
            }

        # Text analysis
        desc_lengths = self.df['text'].str.len()
        word_lengths = self.df['text'].str.split().str.len()

        stats['text'] = {
            'avg_char_length': desc_lengths.mean(),
            'median_char_length': desc_lengths.median(),
            'char_length_std': desc_lengths.std(),
            'avg_word_count': word_lengths.mean(),
            'median_word_count': word_lengths.median(),
            'word_count_std': word_lengths.std(),
        }

        # Most common starting words
        starting_words = self.df['text'].str.split().str[0].str.lower()
        stats['text']['common_starts'] = Counter(starting_words).most_common(10)

        # Image size analysis (from sample)
        image_sizes = self.analyze_image_sizes()
        if image_sizes:
            widths, heights = zip(*image_sizes)
            stats['images'] = {
                'avg_width': np.mean(widths),
                'avg_height': np.mean(heights),
                'common_sizes': Counter(image_sizes).most_common(5)
            }

        return stats

    def print_summary(self, output_file: Optional[Path] = None) -> None:
        """
        Print and optionally save a human-readable summary of the dataset statistics.

        Args:
            output_file: Optional file path to save the summary
        """
        stats = self.analyze()

        # Prepare summary lines
        lines = []
        lines.append("\n=== Dataset Summary ===\n")

        lines.append("General Statistics:")
        lines.append(f"- Total image-text pairs: {stats['general']['total_pairs']:,}")
        lines.append(f"- Total image folders: {stats['general']['total_folders']}")

        if 'labels' in stats:
            lines.append("\nLabel Distribution:")
            for label, count in stats['labels']['distribution'].items():
                lines.append(f"- Label {label}: {count:,} ({count / stats['general']['total_pairs'] * 100:.1f}%)")

        lines.append("\nText Statistics:")
        lines.append(f"- Average description length: {stats['text']['avg_char_length']:.1f} characters")
        lines.append(f"- Average word count: {stats['text']['avg_word_count']:.1f} words")

        lines.append("\nMost common starting words:")
        for word, count in stats['text']['common_starts']:
            lines.append(f"- '{word}': {count:,} times")

        if 'images' in stats:
            lines.append(f"\nImage Size Analysis (from sample):")
            lines.append(
                f"- Average dimensions: {stats['images']['avg_width']:.0f}x{stats['images']['avg_height']:.0f}")
            lines.append("\nMost common sizes:")
            for (w, h), count in stats['images']['common_sizes']:
                lines.append(f"- {w}x{h}: {count} images")

        # Join all lines
        summary = '\n'.join(lines)

        # Print to console
        print(summary)

        # Save to file if specified
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(summary)

    def generate_visualizations(self, output_dir: str) -> None:
        """Generate and save visualization plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Description length distribution
        plt.figure(figsize=(10, 6))
        self.df['text'].str.len().hist(bins=50)
        plt.title('Distribution of Description Lengths')
        plt.xlabel('Number of Characters')
        plt.ylabel('Frequency')
        plt.savefig(output_dir / 'desc_length_dist.png')
        plt.close()

        # Word count distribution
        plt.figure(figsize=(10, 6))
        self.df['text'].str.split().str.len().hist(bins=30)
        plt.title('Distribution of Word Counts')
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.savefig(output_dir / 'word_count_dist.png')
        plt.close()

        if 'label' in self.df.columns:
            # Label distribution
            plt.figure(figsize=(8, 6))
            self.df['label'].value_counts().plot(kind='bar')
            plt.title('Label Distribution')
            plt.xlabel('Label')
            plt.ylabel('Count')
            plt.savefig(output_dir / 'label_dist.png')
            plt.close()