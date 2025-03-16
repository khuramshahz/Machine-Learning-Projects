# Breast Cancer Detection using K-Nearest Neighbors (KNN)

## Overview
This project implements a K-Nearest Neighbors (KNN) model for breast cancer detection using structured data. The model is built using Scikit-Learn and aims to classify instances as malignant or benign.

## Dataset
The dataset used in this project is loaded from Kaggle's `breast-cancer-wisconsin.csv` file. It contains relevant features that help in predicting breast cancer cases.

## Requirements
To run this project, install the following dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Model Architecture
The KNN model works by:
- Calculating the distance between data points
- Identifying the `k` nearest neighbors
- Assigning a class based on majority voting

## Training
The model is trained using the dataset, and hyperparameters such as `k` are optimized for better accuracy.

## Usage
Run the Jupyter Notebook or execute the script to train and evaluate the model:

```bash
python train.py
```

## Results
After training, the model achieves high accuracy in classifying breast cancer cases. The results are visualized using accuracy and confusion matrix plots.



