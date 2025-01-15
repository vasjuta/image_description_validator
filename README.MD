# Image-Text Validation: Baseline Approach

## Dataset Analysis and Baseline Decision

### EDA Findings
Our exploratory data analysis revealed several key characteristics:

1. Dataset Structure:
- 58,000 image-text pairs total
- Significant class imbalance: 84.5% non-matching vs 15.5% matching pairs
- All images standardized to 256x256

2. Text Properties:
- Highly structured descriptions (67.5% start with "a")
- Average length: 10.4 words, 52.5 characters
- Common starting words suggest focus on object quantities ("two", "three", "several")

3. Image Properties:
- Consistent 256x256 dimensions
- Organized in 60 folders (00000-00059)

### Baseline Approach
Based on these findings, we chose the following baseline approach:

1. Model: CLIP-based architecture
- Pretrained CLIP model for joint image-text embedding
- Simple classification head on top
- Reasoning:
  * CLIP is already trained for image-text alignment
  * Handles both modalities efficiently
  * Good at object recognition and counting
  * Minimal preprocessing needed due to standardized data

2. Training Strategy:
- Weighted loss to handle class imbalance (84.5/15.5 ratio)
- Basic train/validation split
- Early stopping on validation F1-score
- No data augmentation in baseline

## Usage

### Data Analysis
```bash
# Run exploratory data analysis
python -m src.utils.eda /path/to/dataset
```

### Training
```bash
# Train the model with default parameters
python scripts/train.py /path/to/dataset

# Train with custom parameters
python scripts/train.py /path/to/dataset \
    --output_dir outputs \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --patience 3
```

### Evaluation
```bash
# Evaluate trained model
python scripts/evaluate.py /path/to/dataset \
    path/to/trained_model.pt \
    --output_dir outputs

# This will produce:
# - metrics.json: Detailed performance metrics
# - error_analysis.csv: Analysis of misclassified examples
# - evaluation.log: Detailed evaluation log
```