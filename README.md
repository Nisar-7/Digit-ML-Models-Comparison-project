# Digit-ML-Models-Comparison-project with PCA on Digits Dataset 
This project applies Principal Component Analysis (PCA) for dimensionality reduction on the Digits dataset and evaluates several machine learning models on the reduced feature set. The PCA step reduces the number of features from 64 to 32, aiming to remove less important information and speed up the training process while maintaining performance.

## Dataset
- Digits dataset from scikit-learn, containing 8x8 images of handwritten digits (0-9).

## Key Steps
1. Load and split the Digits dataset into training and testing sets (70% train, 30% test).  
2. Normalize the features using `MinMaxScaler`.  
3. Apply PCA to reduce feature dimensions from 64 to 32.  
4. Train models on the transformed data.  
5. Evaluate performance using Accuracy, Precision, and Recall metrics.  
6. Visualize and compare model performance using bar charts.

## Models Implemented
- Random Forest Classifier  
- Support Vector Machine (SVM) with linear kernel  
- Artificial Neural Network (ANN) (Multi-layer Perceptron)

## Performance Metrics
- Accuracy  
- Precision (weighted)  
- Recall (weighted)

## Observations
- PCA effectively reduces dimensionality with minimal impact on model accuracy.  
- Model performances are quite similar after PCA transformation, indicating that PCA retains important information.  
- Visualization helps compare training accuracies and test accuracies of the models.

## Requirements
- Python 3.x  
- scikit-learn  
- matplotlib  
- numpy

## How to Run
1. Clone the repository.  
2. Install dependencies (`pip install scikit-learn matplotlib numpy`).  
3. Run the main script to train models and see performance results.

---

Feel free to improve this project by trying different numbers of PCA components or experimenting with additional machine learning algorithms!
