# -*- coding: utf-8 -*-
"""streamlit_app

Modified to include tabs for Introduction, Power BI, and Vehicle Price Prediction.
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

# Create tabs
tab1, tab2, tab3 = st.tabs(["Introduction", "Power BI", "Price Predictor"])

# --- Tab 1: Introduction ---
with tab1:
    st.title("Vehicle Price Prediction Project")
    st.subheader("Problem Statement #1: Pricing Model (Regression)")
    st.markdown("""
    **Goal:**  
    Develop a regression model that can predict the optimal pricing of a vehicle based on its characteristics (e.g., year, make, model, mileage).  
    This model aims to provide price estimates or ranges, helping dealerships fine-tune their pricing strategies to maximize sales while maintaining competitiveness.

    **Welcome to the Vehicle Pricing App**  
    Use the tabs above to navigate between the Power BI dashboard and the Price Prediction Tool.
    """)

# --- Tab 2: Power BI ---
with tab2:
    st.title("Power BI Dashboard")
    st.markdown("Explore visual insights using our Power BI report.")
    st.markdown("[Click here to view the Power BI Dashboard](https://your-powerbi-link.com)", unsafe_allow_html=True)
    st.info("Note: Make sure you have access permission to view the dashboard.")

# --- Tab 3: Price Predictor ---
with tab3:
    st.title("Vehicle Price Prediction App")
    st.markdown("Estimate vehicle price using a trained Random Forest model.")

    # --- Input Fields ---
    make = st.selectbox("Make", sorted(make_model_df["make"].unique()))
    filtered_models = make_model_df[make_model_df["make"] == make]["model"].unique()
    
    if len(filtered_models) == 0:
        filtered_models = label_encoders["model"].classes_
    
    model_name = st.selectbox("Model", sorted(filtered_models))
    model_year = st.slider("Model Year", 2000, 2025, 2020)
    mileage = st.number_input("Mileage (in KM)", value=50000)
    fuel_type = st.selectbox("Fuel Type", label_encoders["fuel_type_from_vin"].classes_)
    transmission = st.selectbox("Transmission", label_encoders["transmission_from_vin"].classes_)
    stock_type = st.selectbox("Stock Type", label_encoders["stock_type"].classes_)

    # --- Predict Button ---
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
