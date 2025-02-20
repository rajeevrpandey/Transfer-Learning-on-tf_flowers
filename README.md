# Flower Classification Using Xception Model

## Overview
This project trains a deep learning model to classify images of flowers into five categories using the **TF-Flowers** dataset. It leverages **Xception**, a pretrained convolutional neural network (CNN), with custom layers for classification. The model is fine-tuned and evaluated using various performance metrics.

## Dataset
The project uses the [TF-Flowers dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers), which consists of five flower categories:
- Dandelion
- Daisy
- Tulips
- Sunflowers
- Roses

## Project Workflow
1. **Load & Preprocess Dataset**  
   - The dataset is split into **Training (75%)**, **Validation (15%)**, and **Testing (10%)**.
   - Images are resized to **224x224 pixels** and normalized using Xception preprocessing.

2. **Model Definition**  
   - The **Xception** model (pretrained on ImageNet) is used as a feature extractor.
   - A **Global Average Pooling (GAP)** layer and a **Dense layer** with softmax activation are added for classification.
   - Initially, the base model layers are **frozen**, training only the new classifier layers.

3. **Model Training**  
   - The model is trained using the **Sparse Categorical Crossentropy** loss function.
   - The **SGD optimizer with momentum** is used, along with an **exponential decay learning rate schedule**.
   - **Early stopping** prevents overfitting.

4. **Fine-Tuning**  
   - The base Xception model is unfrozen and fine-tuned with a lower learning rate.
   - The model is retrained for additional epochs to improve accuracy.

5. **Model Evaluation**  
   - The model is evaluated on the test set, reporting accuracy and loss.
   - A **confusion matrix** visualizes misclassifications.
   - **ROC curves** and **AUC scores** are plotted for multi-class classification.

## Results
- **Test Accuracy:** Achieved high accuracy across all classes.
- **Confusion Matrix:** Displays minimal misclassification.
- **ROC Curve Analysis:** 
  - AUC values for all classes are near **1.00**, indicating **excellent classification performance**.
  - The lowest AUC (~0.98) suggests strong model generalization.

## Performance Visualization
### Training & Validation Curves
- Training and validation **loss** and **accuracy** are plotted to monitor learning progress.

### Confusion Matrix
- A heatmap shows model performance for each flower category.

### ROC Curve
- Plots True Positive Rate vs. False Positive Rate for each class.

## Requirements
To run this project, install the following dependencies:

```bash
pip install tensorflow tensorflow-datasets numpy matplotlib seaborn scikit-learn
```

## Running the Project
1. **Clone the repository**:
   ```bash
   git clone https://github.com/rajeevrpandey/Transfer-Learning-on-tf_flowers/flower-classification.git
   cd flower-classification
   ```
2. **Run the Python script**:
   ```bash
   python train_model.py
   ```

## Future Improvements
- Experiment with **data augmentation** for better generalization.
- Implement **different architectures** such as EfficientNet.
- Use **transfer learning with additional datasets** to improve robustness.

## Author
Rajeev Ranjan Pandey
