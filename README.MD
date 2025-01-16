# Image-Text Validation
This project implements a machine learning pipeline for validating whether text descriptions accurately match their corresponding images. Given an image-text pair, the system predicts whether the description is accurate (1) or mismatched (0). The solution combines modern vision-language models with a custom classification approach to achieve robust validation performance.

Key features:
- High-accuracy image-text validation (>93% F1 score on unseen data)
- CLIP-based architecture for robust semantic understanding
- Production-ready inference pipeline
- Comprehensive evaluation suite

## System Design
The project follows a modular design with clear separation of concerns:

```/src```
- ```data/```: Dataset handling and preprocessing
- ```models/```: CLIP-based baseline model implementation
- ```training/```: Training pipeline with early stopping and metrics tracking
- ```evaluation/```: Comprehensive evaluation suite
- ```inference/```: User-friendly API for model usage
- ```utils/```: EDA, logging utilities, shared preprocessing

```scripts/```: Entry points for training, evaluation, and inference

```tests/```: Basic unit tests

## Dataset Analysis and Model Choice

### EDA Findings
Our exploratory data analysis revealed several key characteristics:

1.Dataset Structure:
- 58,000 image-text pairs total
- Significant class imbalance: 84.5% non-matching vs 15.5% matching pairs
- All images standardized to 256x256

2.Text Properties:
- Highly structured descriptions (67.5% start with "a")
- Average length: 10.4 words, 52.5 characters
- Common starting words suggest focus on object quantities ("two", "three", "several")

3.Image Properties:
- Consistent 256x256 dimensions
- Organized in 60 folders (00000-00059)

### Baseline Approach & Design Choices
### Model Architecture:
- CLIP-based backbone with classification head
- Features:
  * Image and text embeddings from CLIP
  * Concatenated feature vector (CLIP embeddings + similarity score)
  * Multi-layer classification head with dropout
- Reasoning:
  * CLIP provides strong pretrained image-text understanding
  * Architecture handles both exact matches and semantic similarity
  * Efficient for both training and inference
- Model Profile:
  * Total parameters: 88M
  * Trainable parameters: 66 (classification head only)
  * Model size: 584MB
  
### Preprocessing Pipeline:
- Images:
  * Standardized resizing to 256x256
  * Tensor conversion via torchvision transforms
  * Modular design in preprocessing.py for reusability
- Text:
  * Leverages CLIP's built-in processor for tokenization
  * Ensures alignment with CLIP's pretraining expectations
  * Handles padding, truncation, and special tokens automatically

### Training Strategy:
- Weighted BCE loss to handle 84.5/15.5 class imbalance
- Learning rate: 1e-4 with AdamW optimizer
- Early stopping on validation F1-score
- Frozen CLIP weights to prevent catastrophic forgetting

## Baseline Model Performance

### Data Split
- Training set: 40,600 samples (70%)
- Validation set: 8,700 samples (15%)
- Test set: 8,700 samples (15%)

### Training Performance
Training time (6 epochs, batch size 64) on Tesla T4 in Google Colab: ~60 mins

Best model achieved (after 3 epochs):
- Training Loss: 0.1184
- Training F1: 0.9184
- Validation Loss: 0.1098
- Validation F1: 0.9408

Other metrics:
- Training Precision: 0.8606
- Training Recall: 0.9846
- Validation Precision: 0.9114
- Validation Recall: 0.9722 

### Test Set Performance
Performance on unseen data:
- Accuracy: 0.9786
- Precision: 0.8982
- Recall: 0.9680
- F1 Score: 0.9318
- Specificity: 0.9805
- ROC AUC: 0.9961

### Model Considerations
Error analysis on the test set revealed that the model is:
- Most confident on straightforward object/action descriptions
- Reliable at catching completely mismatched content
- More uncertain on abstract or metaphorical descriptions
- Shows appropriate uncertainty (confidence ~0.5) in ambiguous cases

However, the model's performance (93% F1-score on unseen data) already meets accuracy requirements for the scope of this project. 
We chose not to explore alternatives at this stage. See [Potential Improvements](#potential-improvements) section for future enhancement ideas.

## External Dependencies
- ```transformers```: CLIP model and tokenizer (Hugging Face)
  * Chosen for robust pretrained models and active maintenance
- ```torch```: Deep learning framework
  * Industry standard with good GPU support
- ```Pillow```: Image processing
  * Required for image loading and preprocessing
- ```scikit-learn```: Metrics calculation
  * Provides comprehensive evaluation metrics
- ```pandas```: Data handling and analysis
  * Efficient for large dataset management
 
## Usage

### Data Analysis
```bash
# Run exploratory data analysis
python scripts/run_eda.py /path/to/dataset
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

### Inference
```bash
# Command-line inference
python scripts/inference_client.py \
    path/to/model.pt \
    path/to/image.jpg \
    "image description"

# API Usage
from src.inference.api import ImageTextValidator

validator = ImageTextValidator("path/to/model.pt")
result = validator.validate("path/to/image.jpg", "image description")
```

### Potential Improvements And Enhancements
## Model
1. Error Analysis
2. Data augmentation for better generalization
3. Experiment with unfreezing CLIP layers
4. Ensemble approaches combining different CLIP variants
5. Alternative Approaches:
   - Two-tower architecture with separate encoders:
     * Use BERT for text and ResNet/ViT for images instead of CLIP
     * Pros: More control over feature extraction and model behavior
     * Cons: Would require more training data and compute; might not match CLIP's robust pretraining
   - Generation-validation hybrid:
     * Use image captioning model (e.g., BLIP) to generate descriptions
     * Compare semantic similarity between generated and provided captions
     * Pros: Could catch subtle mismatches CLIP might miss
     * Cons: Significantly higher latency (two model passes); larger compute requirements

## System
1. Inference optimization for production deployment
2. Preprocessing pipeline improvements:
   - Extract transform logic to a configurable component
   - Store and version preprocessing configurations with model artifacts
   - Add validation to ensure inference-time transforms match training
3. More thorough error handling and logging throughout the codebase
4. Model versioning and registry
5. Containerization 
6. Cache layer
7. Deal with all training/evaluation/inference warnings

## User Experience
1. Web client
2. Validation progress bar
3. Enable texts (captions) in non-English languages
4. Batch image-text validation
5. Suggest correct text in case of validation miss
6. Censorship for both images and texts
