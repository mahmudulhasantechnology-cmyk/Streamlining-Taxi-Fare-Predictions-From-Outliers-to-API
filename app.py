import streamlit as st
import joblib
import numpy as np
import time

# 1. Page Configuration
st.set_page_config(page_title="Taxi Fare Predictor", page_icon="🚕", layout="centered")

# 2. Injecting Custom White Light-Mode UI Styling via CSS
st.markdown("""
    <style>
        /* Force light theme, white background, and dark text */
        .stApp {
            background-color: #ffffff !important;
            color: #1e293b !important;
        }
        
        /* Tighten global top padding to prevent scrolling */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 1rem !important;
        }
        
        /* Main Title styling - smaller and cleaner */
        .main-title {
            font-size: 2.2rem !important;
            font-weight: 800 !important;
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1.5rem !important;
            text-align: center;
        }
        
        /* Custom High-Conversion "Growth Green" Primary Button Styling */
        div.stButton > button[kind="primary"] {
            background-color: #10b981 !important;
            color: #ffffff !important;
            border: none !important;
            font-weight: 700 !important;
            transition: all 0.2s ease-in-out !important;
        }
        
        /* Smooth hover state for the action button */
        div.stButton > button[kind="primary"]:hover {
            background-color: #059669 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2) !important;
        }
        
        /* Compact light-mode high-impact metric display */
        .metric-container {
            text-align: center;
            padding: 15px;
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, rgba(217, 119, 6, 0.08) 100%);
            border: 1px solid rgba(245, 158, 11, 0.3);
            border-radius: 10px;
            margin-top: 15px;
        }
        .metric-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #b45309;
            font-weight: 700;
        }
        .metric-value {
            font-size: 2.8rem;
            font-weight: 900;
            color: #1e293b;
            margin: 2px 0;
        }
        
        /* Fix label contrast for light mode text inside standard components */
        label, .stSlider, p, span {
            color: #1e293b !important;
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

# 4. Header Section (Clean title with no tagline)
st.markdown('<h1 class="main-title">🚕 Outlier-Resistant Fare Predictor</h1>', unsafe_allow_html=True)

# 5. Interactive Form Card (Using native components to fix the blank box bug)
with st.container(border=True):
    st.markdown("### 📋 Configure Your Journey")
    
    trip_distance = st.slider(
        "Select Trip Distance (in miles):", 
        min_value=0.1, 
        max_value=50.0, 
        value=2.5, 
        step=0.1,
        help="Drag to adjust the target distance for fare estimation."
    )
    
    calculate_btn = st.button("Calculate Predicted Fare →", type="primary", use_container_width=True)


# 6. Live Prediction & High-Impact Output Area
if calculate_btn:
    with st.spinner("Computing robust fare..."):
        time.sleep(0.3) 
        
        input_data = np.array([[trip_distance]])
        prediction = model.predict(input_data)
        
        pred_array = np.asarray(prediction).flatten()
        if pred_array.size > 0:
            fare_value = float(pred_array[0])
        else:
            fare_value = 0.0
    
    # Custom high-impact results widget
    st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Estimated Robust Fare</div>
            <div class="metric-value">${fare_value:.2f}</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.toast("✨ Fare successfully computed!", icon="✅")
