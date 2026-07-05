import streamlit as st
import joblib
import numpy as np

# 1. Page Configuration and Styling
st.set_page_config(page_title="Taxi Fare Predictor", page_icon="🚕", layout="centered")

st.title("🚕 Outlier-Resistant Taxi Fare Predictor")
st.write("This application uses a robust RANSAC Linear Regression model to calculate taxi fares, completely ignoring historical data anomalies.")

# 2. Safely Load your Saved Model
@st.cache_resource
def load_model():
    return joblib.load('ransac_taxi_model.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ Error: 'ransac_taxi_model.pkl' not found. Please ensure the model file is in the same directory as this app script.")
    st.stop()

# 3. Build Input Elements for the user
st.subheader("📋 Enter Ride Details")

# Adjust these inputs based on what feature 'x' originally represented in your notebook
# If 'x' was a single feature (like trip distance), keep this single input slider:
trip_distance = st.slider("Trip Distance (in miles):", min_value=0.1, max_value=50.0, value=2.5, step=0.1)

# 4. Handle Live Prediction
if st.button("Calculate Predicted Fare", type="primary"):
    # Convert input value to a 2D array shape matching sklearn requirements [[value]]
    input_data = np.array([[trip_distance]])
    
    # Run the model
    prediction = model.predict(input_data)
    
    # Safely flatten and extract the float value to prevent TypeErrors
    fare_value = float(np.ravel(prediction)[0])
    
    # Display the final output safely formatted as currency
    st.success(f"### Calculated Fare Estimate: **${fare_value:.2f}**")
    st.caption("ℹ️ Note: This estimation filters out data spikes caused by unusual peak-hour traffic jams or bad GPS tracking glitches.")
