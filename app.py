import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------------------------------------
# Load Model
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")  # loads your xgb1 model
    return model

xgb1 = load_model()

# -----------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------
st.set_page_config(page_title="Car Price Prediction App", page_icon="🚗")
st.title("🚗 Car Price Prediction App")
st.write("This app predicts used car prices using your trained XGBoost model (`xgb1`).")

# -----------------------------------------------------------
# Input Section
# -----------------------------------------------------------
st.header("🧾 Enter Car Details")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Manufacture Year", min_value=1990, max_value=2025, value=2015)
    make = st.text_input("Make (e.g., Toyota, Ford, BMW)", "Toyota")
    model_name = st.text_input("Model Name", "Corolla")
    trim = st.text_input("Trim (e.g., LX, EX, SE)", "LX")
    condition = st.number_input("Condition (1–5)", min_value=1, max_value=5, value=3)
    cylinders = st.number_input("Cylinders", min_value=2, max_value=12, value=4)
    body = st.text_input("Body Type (e.g., Sedan, SUV)", "Sedan")

with col2:
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    odometer = st.number_input("Odometer (in km)", min_value=0, max_value=500000, value=50000)
    color = st.text_input("Car Color", "White")
    interior = st.text_input("Interior Color", "Black")
    seller = st.text_input("Seller Type (Dealer/Private)", "Dealer")
    mmr = st.number_input("MMR Value", min_value=0, value=10000)

sale_year = st.number_input("Sale Year", min_value=2000, max_value=2025, value=2024)
sale_month = st.slider("Sale Month", 1, 12, 6)
sale_day = st.slider("Sale Day", 1, 31, 15)
sale_weekday = st.selectbox("Sale Weekday (0=Mon, 6=Sun)", list(range(7)))

# -----------------------------------------------------------
# Prediction
# -----------------------------------------------------------
if st.button("Predict Car Price 💰"):
    input_data = pd.DataFrame({
        "year": [year],
        "make": [make],
        "model": [model_name],
        "trim": [trim],
        "body": [body],
        "transmission": [transmission],
        "state": [np.nan],  # If you don't have state info, use NaN
        "condition": [condition],
        "odometer": [odometer],
        "color": [color],
        "interior": [interior],
        "seller": [seller],
        "mmr": [mmr],
        "sale_year": [sale_year],
        "sale_month": [sale_month],
        "sale_day": [sale_day],
        "sale_weekday": [sale_weekday]
    })

    # Convert categorical columns to category dtype
    cat_cols = ["make", "model", "transmission", "trim", "body", "state", "color", "interior", "seller"]
    for col in cat_cols:
        input_data[col] = input_data[col].astype("category")

    try:
        prediction = xgb1.predict(input_data)[0]
        st.success(f"💰 Estimated Car Price: ₹ {prediction:,.2f}")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# -----------------------------------------------------------
# Footer
# -----------------------------------------------------------
st.write("---")
st.caption("Built with ❤️ using Streamlit and XGBoost")
