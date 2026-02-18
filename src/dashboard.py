import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="GuardianPay Fraud Radar", page_icon="🛡️")

st.title("🛡️ GuardianPay Fraud Radar")
st.markdown("Real-time transaction monitoring powered by **Feast** and **MLflow**.")

# Input for User ID
user_id = st.number_input("Enter User ID to scan:", min_value=1000, max_value=2000, value=1005)

if st.button("Run Security Scan"):
    with st.spinner("Fetching features and running model..."):
        # CALL YOUR FASTAPI SERVER
        response = requests.get(f"http://localhost:8000/predict/{user_id}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Create nice visual layout
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Security Status", data["status"])
            with col2:
                st.metric("Risk Probability", data["risk_score"])
            
            # Show visual alert
            if data["is_fraud"]:
                st.error(f"🚨 ALERT: Potential fraud detected for User {user_id}!")
            else:
                st.success(f"✅ Transaction verified for User {user_id}.")
        else:
            st.error("Could not connect to the Prediction API. Is main.py running?")

st.divider()
st.caption("GuardianPay MLOps Pipeline v1.0 | Fast-API Port: 8000 | Streamlit Port: 8501")