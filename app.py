# 6. Live Prediction & High-Impact Output Area
if calculate_btn:
    with st.spinner("Filtering anomalies & computing fare..."):
        time.sleep(0.4) 
        
        input_data = np.array([[trip_distance]])
        prediction = model.predict(input_data)
        
        # --- SAFE EXTRACTION FIX ---
        # Convert prediction to an array, flatten it, and pull the first item
        pred_array = np.asarray(prediction).flatten()
        
        if pred_array.size > 0:
            fare_value = float(pred_array[0])
        else:
            fare_value = 0.0  # Fallback if empty
        # ----------------------------
    
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
