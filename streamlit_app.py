# -*- coding: utf-8 -*-
"""streamlit_app

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1A1D-ugVB9hdvGFny6VKRogmBYe-HIa7e
"""

import streamlit as st
import pandas as pd
import pickle
import gdown

# Download the Random Forest model from Google Drive
model_url = "https://drive.google.com/uc?id=1AIHEGkrnY3mtmOjmMlXEZ4SAAxjaYig2"
output_path = "rf_model.pkl"
gdown.download(model_url, output_path, quiet=False)

# Load the trained model
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the label encoders
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Load make-model pairs CSV (you must upload this to your GitHub repo)
make_model_df = pd.read_csv("make_model_pairs.csv")

# App Title
st.title("Vehicle Price Prediction App")
st.markdown("Estimate vehicle price using a trained Random Forest model.")

# --- Input Fields ---

# Select Make from dataset
make = st.selectbox("Make", sorted(make_model_df["make"].unique()))

# Filter models for the selected make
filtered_models = make_model_df[make_model_df["make"] == make]["model"].unique()

# If nothing matches (edge case), fall back to full model list
if len(filtered_models) == 0:
    filtered_models = label_encoders["model"].classes_

# Select Model based on filtered list
model_name = st.selectbox("Model", sorted(filtered_models))

# Other inputs
model_year = st.slider("Model Year", 2000, 2025, 2020)
mileage = st.number_input("Mileage (in KM)", value=50000)
fuel_type = st.selectbox("Fuel Type", label_encoders["fuel_type_from_vin"].classes_)
transmission = st.selectbox("Transmission", label_encoders["transmission_from_vin"].classes_)
stock_type = st.selectbox("Stock Type", label_encoders["stock_type"].classes_)

# --- Predict Button ---
if st.button("Predict Price"):
    # Prepare input DataFrame
    input_df = pd.DataFrame({
        "make": [make],
        "model": [model_name],
        "model_year": [model_year],
        "mileage": [mileage],
        "fuel_type_from_vin": [fuel_type],
        "transmission_from_vin": [transmission],
        "stock_type": [stock_type]
    })

    # Encode categorical features using label encoders
    for col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])

    # Predict and show result
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Price: ${prediction:,.2f}")
