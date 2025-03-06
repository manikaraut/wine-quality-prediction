import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved models
knn_with_pca = joblib.load('C:/Users/Lenovo/projects/machine_learning/models/wine_classifier.pkl')
scaler = joblib.load('C:/Users/Lenovo/projects/machine_learning/models/scalar.pkl')
pca = joblib.load('C:/Users/Lenovo/projects/machine_learning/models/pca.pkl')

# Function to predict wine quality
def predict_wine_quality(data):
    # Step 1: Scale the new data
    data_scaled = scaler.transform([data])

    # Step 2: Apply PCA transformation
    data_pca = pca.transform(data_scaled)

    # Step 3: Predict using KNN classifier
    prediction = knn_with_pca.predict(data_pca)

    return prediction[0]

# Streamlit UI for the user to input wine data
def main():
    st.title("🍷 Wine Quality Prediction App")

    # Input fields for the user to enter wine data
    st.header("Enter the wine characteristics:")

    fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=20.0, value=7.4, step=0.1)
    volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=1.5, value=0.3, step=0.01)
    citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=20.0, value=6.0, step=0.1)
    chlorides = st.number_input("Chlorides", min_value=0.0, max_value=0.2, value=0.04, step=0.001)
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=200.0, value=100.0, step=1.0)
    density = st.number_input("Density", min_value=0.98, max_value=1.05, value=0.99, step=0.001)
    pH = st.number_input("pH", min_value=2.5, max_value=4.0, value=3.2, step=0.01)
    sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.0, value=0.6, step=0.01)
    alcohol = st.number_input("Alcohol", min_value=8.0, max_value=15.0, value=10.0, step=0.1)

    # Collect user input into a list
    user_input = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                  free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]

    # Button to make the prediction
    if st.button("Predict Wine Quality"):
        # Predict wine quality using the loaded model
        predicted_quality = predict_wine_quality(user_input)

        # Display the result
        st.write(f"The predicted wine quality category is: **{predicted_quality}**")

if __name__ == "__main__":
    main()
