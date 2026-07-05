import streamlit as st
import joblib
import numpy as np
import time

# 1. Page Configuration
st.set_page_config(page_title="Taxi Fare Predictor", page_icon="🚕", layout="centered")

# 2. Injecting Custom "Punchy" UI Styling via CSS
st.markdown("""
    <style>
        /* Main background and global text color tweaks */
        .stApp {
            background-color: #0d0f12;
            color: #e2e8f0;
        }
        
        /* Main Title styling */
        .main-title {
            font-size: 2.8rem !important;
            font-weight: 800 !important;
            background: linear-gradient(135deg, #ffbd03 0%, #ff6b00 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        /* Sub-tagline styling */
        .tagline {
            font-size: 1.1rem;
            color: #94a3b8;
            margin-bottom: 2rem;
            line-height: 1.6;
        }
        
        /* Custom card wrapper for input & results */
        .premium-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        }
        
        /* Large high-impact metric display */
        .metric-container {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, rgba(255, 189, 3, 0.1) 0%, rgba(255, 107, 0, 0.1) 100%);
            border: 1px solid rgba(255, 189, 3, 0.3);
            border-radius: 12px;
            margin-top: 15px;
        }
        .metric-label {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: #ffbd03;
            font-weight: 700;
        }
        .metric-value {
            font-size: 3.5rem;
            font-weight: 900;
            color: #ffffff;
            margin: 5px 0;
            text-shadow: 0 0 15px rgba(255, 189, 3, 0.3);
        }
    </style>
""", unsafe_allow_html=True)

# 3. Model Loading
@st.cache_resource
def load_model():
    return joblib.load('ransac_taxi_model.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ Error: 'ransac_taxi_model.pkl' not found.")
    st.stop()

# 4. Header Section (Using custom CSS classes)
st.markdown('<h1 class="main-title">🚕 Outlier-Resistant Fare Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="tagline">An intelligent taxi fare engine backed by RANSAC regression—engineered to automatically strip away historical traffic anomalies, surge glitches, and data spikes.</p>', unsafe_allow_html=True)

# 5. Interactive Form Card
# Open the styled card wrapper
st.markdown('<div class="premium-card">', unsafe_allow_html=True)
st.markdown("### 📋 Configure Your Journey")

trip_distance = st.slider(
    "Select Trip Distance (in miles):", 
    min_value=0.1, 
    max_value=50.0, 
    value=2.5, 
    step=0.1,
    help="Drag to adjust the target distance for fare estimation."
)

# Crucial: Define the variable globally on the root page
calculate_btn = st.button("Calculate Predicted Fare →", type="primary", use_container_width=True)

# Close the styled card wrapper
st.markdown('</div>', unsafe_allow_html=True)


# 6. Live Prediction & High-Impact Output Area
if calculate_btn:
    with st.spinner("Filtering anomalies & computing fare..."):
        time.sleep(0.4) 
        
        input_data = np.array([[trip_distance]])
        prediction = model.predict(input_data)
        
        # Convert prediction to an array, flatten it, and pull the first item safely
        pred_array = np.asarray(prediction).flatten()
        
        if pred_array.size > 0:
            fare_value = float(pred_array[0])
        else:
            fare_value = 0.0  # Fallback if empty
    
    # Custom high-impact results widget
    st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Estimated Robust Fare</div>
            <div class="metric-value">${fare_value:.2f}</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.toast("✨ Fare successfully computed!", icon="✅")
    
    st.markdown("")
    st.caption("ℹ️ **Anomaly Filter Active:** This specific estimation filters out structural pricing spikes caused by extreme weather, sudden airport gridlocks, or historical GPS telemetry tracking errors.")
