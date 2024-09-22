# AnomaData Project

## Project Overview
This data science project aims to classify anomalies in a given dataset using machine learning models. The project performs data preprocessing, model training, and evaluation to create a robust classification model.

## Project Structure
The project is organized into the following structure:

/AnomaData
├── README.md              # Detailed instructions for setting up and running the project
├── app.py                  # Flask script to deploy the project (if required)
├── data/
│   ├── AnomaData.xlsx       # The original dataset
│   └── AnomaData.csv        # The dataset in CSV format (converted for easier use)
├── models/
│   └── best_model.pkl       # Saved trained model file(auto-generated)
├── notebook/
│   └── anomaData.ipynb  # Jupyter notebook containing the entire workflow
├── requirements.txt         # File containing all the dependencies needed to run the project
├── visuals/
    ├── features.png         # Histogram visualization of features(auto-generated)
    └── roc_curve.png        # ROC curve visualization of the model(auto-generated)



## Problem Statement
The goal of this project is to identify anomalies in the dataset by applying machine learning techniques. The dataset consists of multiple features, and the target variable `y` represents the class labels.

## Dataset
The dataset is stored in the `/data/` directory:
- **AnomaData.xlsx**: Original Excel dataset.
- **AnomaData.csv**: The dataset in CSV format for easier usage.

## Model
The model used for classification is **XGBoost**. The model has been trained and saved as `best_model.pkl` in the `/models/` directory. Hyperparameter tuning was performed using **RandomizedSearchCV**.

## Installation & Setup Instructions
1. Clone the repository:
    ```bash
    git clone https://github.com/Mosesomo/AnomaDataScience
    cd AnomaDataScience
    ```

2. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Open the Jupyter Notebook in your preferred environment (VS Code, Jupyter Lab, etc.):
    ```bash
    jupyter notebook notebook/anomaData.ipynb
    ```

4. (Optional) If using the Flask app, run the app with:
    ```bash
    python3 app.py
    ```

## Project Workflow
1. **Data Loading and Exploration**: Load and explore the data to understand the features and handle missing values or anomalies.
2. **Data Preprocessing**: Remove outliers and handle class imbalance using techniques like **SMOTE** and **RandomUnderSampler**.
3. **Model Training**: Train the **XGBoost Classifier** using hyperparameter tuning to optimize the model's performance.
4. **Model Evaluation**: Evaluate the model using classification metrics such as **confusion matrix**, **ROC-AUC score**, and **classification report**.
5. **Visualizations**: Generate and save feature histograms and ROC curve plots.

## Dependencies
Make sure to have the following libraries installed (in `requirements.txt`):
- pandas
- numpy
- matplotlib
- scikit-learn
- imbalanced-learn
- xgboost
- openpyxl
- jupyter

## Results
- **Model Performance**: Achieved a high ROC-AUC score, demonstrating good classification performance.
- **Visualizations**: Feature histograms and ROC curve are saved in the `/visuals/` directory.

