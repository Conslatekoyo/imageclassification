
# Image Classification Project

A machine learning project to classify images using multiple algorithms, comparing the performance of each model on the dataset. The project covers data preprocessing, model training, evaluation, and visualization through confusion matrices.

## Introduction

This project involves building and evaluating four different machine learning models to classify images into categories. Each model's performance is assessed using accuracy metrics and confusion matrices to understand how well it performs on the test dataset. The models explored are:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Multilayer Perceptron (MLP) Neural Network

## Dataset

The dataset contains images of \(28 \times 28\) pixels, commonly used in classification tasks. It is split into:
- **Training Set**: Used to train the models.
- **Test Set**: Used to evaluate model performance.

## Project Structure

```
.
├── data/
│   ├── images.csv                 # Dataset (to be loaded)
├── notebooks/
│   ├── Image_Classification.ipynb  # Jupyter notebook with implementation
├── src/
│   ├── preprocess.py               # Script for data preprocessing
│   ├── models.py                   # Model definitions and training scripts
├── README.md                       # Project README file
└── requirements.txt                # Python dependencies
```

## Requirements

To run this project, you’ll need the following Python libraries:
- `numpy`
- `scikit-learn`
- `tensorflow` (for neural network model)
- `matplotlib`

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## Data Preprocessing

Data preprocessing is essential for optimal model performance. The following steps were applied:
1. **Flattening**: Each image was reshaped into a 1D array for compatibility with the models.
2. **Normalization**: Pixel values were scaled to [0, 1] for neural network training.
3. **Train-Test Split**: Data was split into training and test sets if not already split.

## Machine Learning Models

The following models were trained and evaluated:
- **Logistic Regression**: A linear model commonly used for classification tasks.
- **K-Nearest Neighbors (KNN)**: A non-parametric algorithm that classifies based on the closest training examples.
- **Support Vector Machine (SVM)**: A supervised learning model that finds the optimal hyperplane for classification.
- **Multilayer Perceptron (MLP) Neural Network**: A basic neural network model with one hidden layer.

### Example Code for Training MLP

```python
from tensorflow.keras import layers, models

mlp_model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),  # First hidden layer
    layers.Dense(10, activation='softmax')  # Output layer
])

mlp_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

mlp_model.fit(x_train_flat, y_train, epochs=10, validation_split=0.2)
```

## Evaluation Metrics

Each model was evaluated using the following metrics:
- **Accuracy**: The percentage of correct predictions.
- **Confusion Matrix**: Displays the number of true positives, false positives, true negatives, and false negatives for each class, helping visualize misclassifications.

### Generating a Confusion Matrix

Example for MLP Model:
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Get predictions and calculate confusion matrix
y_pred = np.argmax(mlp_model.predict(x_test_flat), axis=1)
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(cmap='Reds')
```

## Results

- **Logistic Regression**: Accuracy: 92.46%
- **K-Nearest Neighbors**: Accuracy: 97.05%
- **Support Vector Machine**: Accuracy: 94.04%
- **Multilayer Perceptron (MLP)**: Accuracy: 97.29%

Based on the model performance metrics, here are the key observations:

- **Multilayer Perceptron (MLP)** achieved the highest accuracy at **97.29%**, indicating that this neural network model could capture complex patterns in the data, making it highly effective for this classification task.
  
- **K-Nearest Neighbors (KNN)** also performed exceptionally well, achieving an accuracy of **97.05%**. This model's strong performance suggests that similar images tend to cluster well, allowing KNN to make accurate predictions by leveraging proximity in feature space.

- **Support Vector Machine (SVM)** reached an accuracy of **94.04%**, performing well but slightly below KNN and MLP. SVM's performance indicates that a linear or kernel-based decision boundary effectively classifies the data but may miss some of the nuanced patterns captured by neural networks and KNN.

- **Logistic Regression** had the lowest accuracy at **92.46%**. While still quite effective, this model's linear nature likely limits its ability to capture complex relationships in the data compared to the other, more sophisticated models.

### Overall Insights:
- **Non-linear models** (KNN, MLP, and SVM) outperformed **Logistic Regression**, highlighting the importance of model complexity for this dataset.
- **MLP** demonstrated the highest accuracy, suggesting that a neural network architecture is well-suited for this type of image classification task.
- **KNN**'s strong performance without any complex training confirms that the dataset has well-defined clusters, making it a viable option when computational resources are limited.N** struggled due to high-dimensionality, especially without dimensionality reduction.

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/image-classification-project.git
    cd image-classification-project
    ```
2. Prepare the dataset and place it in the `data` folder.

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Jupyter Notebook to train and test models:
    ```bash
    jupyter notebook notebooks/Image_Classification.ipynb
    ```

## Future Improvements

- **Hyperparameter Tuning**: Optimizing parameters for better model accuracy.
- **Data Augmentation**: Increasing dataset size with transformations to improve generalization.
- **Advanced Neural Networks**: Using Convolutional Neural Networks (CNNs) for image data.

## Contributing

Contributions are welcome! Please fork the repository, make changes, and submit a pull request.

