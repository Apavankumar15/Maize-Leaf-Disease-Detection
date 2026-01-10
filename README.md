#  Maize Leaf Disease Detection using Deep Learning

This project implements an end-to-end deep learning approach to automatically detect and classify maize (corn) leaf diseases from images using a Convolutional Neural Network (CNN).

The model classifies maize leaf images into four categories: Blight, Common Rust, Gray Leaf Spot, and Healthy.


##  Problem Statement

Maize leaf diseases significantly reduce crop yield and food security. Traditional disease identification relies on manual inspection by experts, which is time-consuming and not scalable.  
This project aims to automate maize leaf disease detection using deep learning techniques.


##  Dataset

- **Source**: Kaggle – Corn or Maize Leaf Disease Dataset
- **Total Images**: 4,188
- **Classes**:
  - Blight
  - Common Rust
  - Gray Leaf Spot
  - Healthy

The dataset was divided into training, validation, and test sets.



## Methodology

1. Data cleaning and validation
2. Train–Validation–Test split
3. Image preprocessing (resize and normalization)
4. Model training using ResNet-18
5. Model evaluation using accuracy and confusion matrix
6. Prediction on unseen images


##  Model Details

- **Architecture**: ResNet-18
- **Framework**: PyTorch
- **Input Size**: 224 × 224
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam


##  Results

- Training Accuracy: ~99%
- Test Accuracy: ~97–99%
- High precision and recall across all disease classes

The model successfully predicts disease categories for unseen maize leaf images.



##  Project Structure

