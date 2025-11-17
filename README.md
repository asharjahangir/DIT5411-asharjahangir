# DIT5411-asharjahangir

## Project Overview

This project implements an AI-powered Chinese character recognition system using deep learning with TensorFlow and OpenCV. The system is designed to recognize handwritten Chinese characters by training on a dataset of PNG images containing various Chinese characters.

## Project Requirements

### Data Strategy
- **Training Set**: First 40 samples (approximately 80%) of each Chinese character
- **Testing Set**: Remaining samples (approximately 20%) of each character
- **Total Output Classes**: 13,065 neurons in the output layer (one per character class)

### Data Augmentation
Since each character has only 40 training samples, extensive data augmentation is applied to create at least 200 samples per character for training. Augmentation techniques include:
- **Rotation**: Multiple angle variations
- **Shearing**: Distortion transformations
- **Scaling**: Size variations
- **Brightness Adjustment**: Lighting variations
- **Gaussian Blur**: Noise introduction

### Model Architecture
- Multiple convolutional neural network (CNN) architectures are implemented and compared
- Each model features multiple layers with different configurations
- Final output layer contains 13,065 neurons corresponding to each Chinese character class

## Technical Implementation

### Scalable Settings
```
MAX_CHARACTERS = 1000              # Limit characters for testing
AUGMENTED_SAMPLES_PER_CLASS = 50   # Augmented samples per character
IMAGE_SIZE = (64, 64)              # Input image dimensions
EPOCHS = 20                        # Training epochs
BATCH_SIZE = 32                    # Training batch size
```

### For Full Deployment
```
MAX_CHARACTERS = 13065             # All 13,065 characters
AUGMENTED_SAMPLES_PER_CLASS = 200  # Full augmentation
EPOCHS = 50                        # Extended training
```

### Libraries Used
- **Python** - Core programming language
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Image processing and augmentation
- **NumPy** - Numerical computations
- **Scikit-learn** - Data preprocessing and label encoding
- **Matplotlib** - Visualization (if needed)


### Key Features

1. **Robust File Handling**
   - Special implementation to handle Chinese characters in file paths
   - Uses `np.fromfile()` and `cv2.imdecode()` for reliable image loading

2. **Memory-Efficient Processing**
   - Configurable character limit for testing (default: 1000 characters)
   - Chunk-based processing to handle large datasets
   - Progress tracking during data loading and processing

3. **Multiple Model Architectures**
   - **Model 1**: Simple CNN with 3 convolutional layers
   - **Model 2**: Deep CNN with batch normalization
   - **Model 3**: Wider CNN with multiple convolutional blocks

4. **Advanced Training Features**
   - Early stopping to prevent overfitting
   - Learning rate reduction on plateau
   - Automatic model selection based on test accuracy

## File Structure

```
project/
├── assignment.py # Main implementation file
├── sampleimages/ # Directory containing PNG images
│ ├── 丁_0.png
│ ├── 丁_1.png
│ └── ...
└── best_chinese_character_model.h5 # Saved best model as a .h5 file as a dataset
```

## Results
```
=== RESULTS ===
Model 1 (Simple CNN): 0.5450
Model 2 (Deep CNN): 0.0250
Model 3 (Wider CNN): 0.3725

Best model: Model 1 (Simple CNN) with accuracy: 0.5450
Best model saved as 'best_chinese_character_model.h5'
```
