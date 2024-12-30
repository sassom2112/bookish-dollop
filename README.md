# FashionMNIST Classification Project

## Overview
This project builds and trains a Convolutional Neural Network (CNN) to classify images from the FashionMNIST dataset into 10 categories. The FashionMNIST dataset consists of grayscale 28x28 pixel images of clothing and footwear items, categorized into 10 classes such as T-shirts, trousers, sneakers, etc.

## Project Structure
The project includes the following key components:
1. **Data Preparation**:
   - Downloading and transforming the FashionMNIST dataset for training, validation, and testing.
   - Applying data augmentation techniques such as random flipping, rotation, and cropping to improve model robustness.

2. **Model Definition**:
   - A CNN architecture with convolutional, pooling, and dropout layers to enhance performance and reduce overfitting.
   - Fully connected layers for classification into 10 categories.

3. **Training and Evaluation**:
   - Training the CNN using a loss function (CrossEntropyLoss), optimizer (Adam), and learning rate scheduler (ExponentialLR).
   - Evaluating the model's performance on validation and test datasets using metrics such as accuracy and confusion matrix.

4. **Visualization**:
   - Plotting training and validation loss/accuracy curves over epochs.
   - Displaying example images from each class and visualizing the confusion matrix.

## Dataset
The **FashionMNIST** dataset includes:
- **Training Set**: 60,000 images.
- **Test Set**: 10,000 images.
- **Classes**:
  0. T-shirt/top  
  1. Trouser  
  2. Pullover  
  3. Dress  
  4. Coat  
  5. Sandal  
  6. Shirt  
  7. Sneaker  
  8. Bag  
  9. Ankle boot  

## Key Functions and Scripts
### Data Preparation
- **`transform_train` and `transform_test`**:
  Apply data augmentation for training and normalization for testing.
- **`DataLoader`**:
  Load the dataset and create data loaders for training, validation, and testing.

### Model Definition
- **`CNNClassifier`**:
  A PyTorch model with convolutional layers, dropout regularization, and fully connected layers.

### Training and Evaluation
- **`train_classification_model`**:
  Trains the CNN and tracks the best model based on validation accuracy.
- **`plot_training_curves`**:
  Visualizes loss and accuracy over epochs for training, validation, and testing.
- **`plot_cm`**:
  Generates a confusion matrix to evaluate classification performance.

### Visualization
- **`show_class_examples`**:
  Displays one example image from each class in a 2x5 grid.

## Example Outputs
1. **Training Curves**:
   - Plots of training/validation loss and accuracy over epochs.
2. **Class Examples**:
   - Visualizations of one image from each class with labels.
3. **Confusion Matrix**:
   - Visualization of true vs. predicted class labels to analyze performance.

## Requirements
- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn
- TorchVision

## How to Run
1. Clone the repository and install the required libraries.
2. Run the script to download and preprocess the dataset.
3. Train the model using `train_classification_model`.
4. Evaluate the model and visualize the results using the utility functions.

## Future Improvements
- Hyperparameter tuning for better performance.
- Use advanced architectures (e.g., ResNet) for improved accuracy.
- Explore transfer learning with pre-trained models.

## Author
This project was developed to showcase image classification with PyTorch and the FashionMNIST dataset.
