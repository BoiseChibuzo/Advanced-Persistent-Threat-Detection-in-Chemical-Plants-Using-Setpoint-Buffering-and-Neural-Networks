# Advanced Persistent Threat Detection in Chemical Plants Using Setpoint Buffering and Neural Networks

This repository contains the code and instructions to reproduce the results presented in the paper **Advanced Persistent Threat Detection in Chemical Plants Using Setpoint Buffering and Neural Networks**.

## Requirements

To reproduce this work, you need the following software and libraries:

- Python 3.x
- NumPy
- Matplotlib
- Seaborn
- TensorFlow
- Scikit-learn
- Imbalanced-learn

You can install the required libraries using the following command:

```bash
pip install numpy matplotlib seaborn tensorflow scikit-learn imbalanced-learn
```
## Data Generation
The synthetic data for this study simulates temperature and pressure readings in a chemical plant. The data includes both normal and anomalous readings to train and evaluate the model.

## Data Preprocessing
Before training the model, the data needs to be preprocessed:

- **Split Data**: Divide the dataset into training and test sets.
- **Apply SMOTE**: Use Synthetic Minority Over-sampling Technique (SMOTE) to balance the training set.

## Model Training
The neural network model is trained using the balanced dataset.

### Model Architecture:
- **Input Layer**: 4 features (temperature, pressure, and their differences from setpoints).
- **Hidden Layer 1**: 64 neurons (ReLU activation).
- **Dropout Layer 1**: Dropout rate of 0.5.
- **Hidden Layer 2**: 32 neurons (ReLU activation).
- **Dropout Layer 2**: Dropout rate of 0.5.
- **Output Layer**: 1 neuron (Sigmoid activation).

### Hyperparameters:
- **Batch Size**: 32
- **Epochs**: 100
- **Learning Rate**: 0.001
- **Class Weight**: {0: 1.0, 1: 10.0}

## Model Evaluation
The model is evaluated on the test set using various metrics:

- Accuracy
- Precision
- Recall
- F1-score
- PR AUC
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa
- Balanced Accuracy

### Plotting Results:
- **ROC Curve**
- **Confusion Matrix**
- **Learning Curves**
## Reproducing the Results

### Clone the Repository
To reproduce the results, first clone the repository:

```bash
git clone <repository_url>
cd <repository_directory>
```
### Install Requirements
Ensure you have the necessary Python packages by installing the requirements:
```bash
pip install -r requirements.txt
```
### Run the Notebook
Open the Jupyter notebook APT_anomaly_detection.ipynb and run all cells to generate the synthetic data, train the neural network model, and evaluate its performance.

### Visualize Results
Ensure all plots are displayed correctly to verify the model's performance and the effectiveness of the training process. The notebook includes code to plot the following:

ROC Curve: Displays the model's performance in distinguishing between normal and anomalous data points.
Confusion Matrix: Illustrates the number of true positives, false positives, true negatives, and false negatives.
Learning Curves: Show the training and validation accuracy and loss over the epochs.
