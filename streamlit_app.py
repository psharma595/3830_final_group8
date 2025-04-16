# -*- coding: utf-8 -*-
"""streamlit_app

Includes only two tabs: Introduction and Price Predictor (Power BI tab removed).
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

# Load make-model pairs CSV
make_model_df = pd.read_csv("make_model_pairs.csv")

# Create tabs: only Introduction and Price Predictor
intro_tab, predictor_tab = st.tabs(["Introduction", "Price Predictor"])

# --- Introduction Tab ---
with intro_tab:
    st.title("Vehicle Price Prediction Project")

    st.subheader("About the Project")
    st.markdown("""
    In the dynamic world of automotive sales, setting the right price for a vehicle is essential to staying competitive while maximizing revenue.  
    This project introduces a data-driven pricing tool built using machine learning to predict optimal vehicle prices based on characteristics like make, model, year, mileage, fuel type, and more.

    By using a trained regression model, this application helps dealerships generate accurate price estimates, enabling smarter pricing strategies rooted in real market data.
    """)

    st.subheader("How to Use This App")
    st.markdown("""
    This interactive web application consists of two main sections:

    - **Introduction:** Learn about the objective and functionality of the project.
    - **Price Predictor Tool:** Input vehicle details and receive an instant estimated price using machine learning.
    """)

# --- Price Predictor Tab ---
with predictor_tab:
    st.title("Vehicle Price Prediction App")
    st.markdown("Estimate a vehicle's price using our trained Random Forest regression model.")

    # Input Fields
    make = st.selectbox("Make", sorted(make_model_df["make"].unique()))
    filtered_models = make_model_df[make_model_df["make"] == make]["model"].unique()
    
    if len(filtered_models) == 0:
        filtered_models = label_encoders["model"].classes_

    model_name = st.selectbox("Model", sorted(filtered_models))
    model_year = st.slider("Model Year", 2014, 2024, 2017)
    mileage = st.number_input("Mileage (in KM)", value=50000)
    fuel_type = st.selectbox("Fuel Type", label_encoders["fuel_type_from_vin"].classes_)
    transmission = st.selectbox("Transmission", label_encoders["transmission_from_vin"].classes_)
    stock_type = st.selectbox("Stock Type", label_encoders["stock_type"].classes_)

    # Prediction Button
    if st.button("Predict Price"):
        input_df = pd.DataFrame({
            "make": [make],
            "model": [model_name],
            "model_year": [model_year],
            "mileage": [mileage],
            "fuel_type_from_vin": [fuel_type],
            "transmission_from_vin": [transmission],
            "stock_type": [stock_type]
        })

        for col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])

        prediction = model.predict(input_df)[0]
        st.success(f"Estimated Price: ${prediction:,.2f}")
