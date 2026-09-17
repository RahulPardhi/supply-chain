
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model, scaler, and label encoder
model = joblib.load('best_classification_model.joblib')
scaler = joblib.load('scaler_clf.joblib')
label_encoder = joblib.load('label_encoder.joblib')
feature_names = joblib.load('feature_names.joblib')

st.set_page_config(page_title="Supply Chain Risk Prediction", layout="wide")

st.title("Supply Chain Risk Classification")
st.write("Enter the feature values below to predict the risk classification.")

# Create input fields for each feature
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(f"Enter value for {feature.replace('_', ' ').title()}:", value=0.0)

# Predict button
if st.button("Predict Risk Classification"):
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure the order of columns matches the training data
    input_df = input_df[feature_names]

    # Scale the input features
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction_encoded = model.predict(scaled_input)

    # Inverse transform the prediction to get original label
    prediction_label = label_encoder.inverse_transform(prediction_encoded)

    st.success(f"The predicted Supply Chain Risk Classification is: **{prediction_label[0]}**")

