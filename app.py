# app.py
import streamlit as st
import pandas as pd
import numpy as np
from model import load_data, load_model

st.title("Laptop Price Prediction App")

# Load data and model
data = load_data()
model = load_model()  # Load the pre-trained model

st.header("Enter Laptop Details for Price Prediction")

# User inputs
company = st.selectbox("Company", data['Company'].unique())
type_name = st.selectbox("Type", data['TypeName'].unique())
inches = st.slider("Screen Size (in Inches)", 10.0, 20.0, 15.6)
cpu = st.selectbox("CPU", data['Cpu'].unique())
ram = st.selectbox("RAM (GB)", sorted(data['Ram'].unique()))
gpu = st.selectbox("GPU", data['Gpu'].unique())
os = st.selectbox("Operating System", data['OpSys'].unique())
weight = st.number_input("Weight (in Kg)", min_value=0.5, max_value=5.0, value=1.5)
width = st.number_input("Screen Width (in Pixels)", min_value=800, max_value=4000, value=1920)
height = st.number_input("Screen Height (in Pixels)", min_value=600, max_value=3000, value=1080)
has_ssd = st.selectbox("Has SSD?", ["Yes", "No"])
has_hdd = st.selectbox("Has HDD?", ["Yes", "No"]) 

# Create input dictionary for prediction
user_input = {
    'Company': company,
    'TypeName': type_name,
    'Inches': inches,
    'Cpu': cpu,
    'Ram': ram,
    'Gpu': gpu,
    'OpSys': os,
    'Weight': weight,
    'Width': width,
    'Height': height,
    'Has_SSD': 1 if has_ssd == "Yes" else 0,
    'Has_HDD': 1 if has_hdd == "Yes" else 0
}

# Predict the price when the button is clicked
if st.button("Predict Price"):
    predicted_price_log = model.predict(pd.DataFrame([user_input]))[0]
    predicted_price = np.expm1(predicted_price_log)  # Convert back from log scale
    st.success(f"Predicted Laptop Price: ₹{predicted_price:,.2f}")
