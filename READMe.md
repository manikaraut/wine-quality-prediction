# Wine Quality Prediction

## Project Overview
This project aims to predict the quality of wine based on its chemical composition. It uses machine learning techniques to classify wine into different quality categories.

## Features
- Interactive **Streamlit Web App** for user-friendly predictions.
- **Pre-trained Machine Learning Model** for fast and accurate classification.
- **Scikit-Learn & Joblib** for model handling.
- **User Input Form** for entering wine properties.

## Dataset
The dataset includes various chemical attributes of wine, such as:
- **Fixed Acidity**
- **Volatile Acidity**
- **Citric Acid**
- **Residual Sugar**
- **Chlorides**
- **Free Sulfur Dioxide**
- **Total Sulfur Dioxide**
- **Density**
- **pH**
- **Sulphates**
- **Alcohol**
- **Quality Category** (Target Variable)

## Installation & Setup
### 1️. Install Dependencies
```bash
pip install streamlit scikit-learn pandas numpy joblib, matplotlib, seaborn
```

### 2️. Run the App
```bash
streamlit run app.py
```

## 3. Usage
1. Open the Streamlit web app.
2. Enter the wine characteristics in the form fields.
3. Click on **Predict Quality** to get the classification.

## Model Details
- **Algorithm Used**: KNN and PCA 
- **Preprocessing**: Feature scaling using StandardScaler
- **Training Data**: High/ Medium/ Low  dataset

## Files & Directories
- `app.py` → Streamlit app for wine quality prediction
- `wine_model.pkl` → Trained ML model
- `scaler.pkl` → StandardScaler for input normalization
- `README.md` → Documentation


