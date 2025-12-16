import streamlit as st
import numpy as np
import joblib

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="Delhi Air Quality Predictor",
    page_icon="🌫️",
    layout="centered"
)

# ----------------------------------
# Load Model
# ----------------------------------
model = joblib.load("rf_model.joblib")

# ----------------------------------
# Title
# ----------------------------------
st.title("🌫️ Delhi Air Quality Predictor")
st.write(
    "Enter current pollution levels to predict **PM2.5 for tomorrow** and assess air safety."
)

# ----------------------------------
# Input Section
# ----------------------------------
st.subheader("🧪 Enter Pollution Levels")

co = st.number_input("CO", 0.0, 5000.0, 500.0)
no = st.number_input("NO", 0.0, 500.0, 20.0)
no2 = st.number_input("NO₂", 0.0, 500.0, 40.0)
o3 = st.number_input("O₃", 0.0, 500.0, 30.0)
so2 = st.number_input("SO₂", 0.0, 500.0, 10.0)
pm10 = st.number_input("PM10", 0.0, 1000.0, 120.0)
nh3 = st.number_input("NH₃", 0.0, 500.0, 20.0)

st.markdown("**PM2.5 History**")
pm25_lag1 = st.number_input("PM2.5 (1 hour ago)", 0.0, 1000.0, 100.0)
pm25_lag24 = st.number_input("PM2.5 (24 hours ago)", 0.0, 1000.0, 100.0)

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("Predict Air Quality 🚀"):

    input_data = np.array([[
        co, no, no2, o3, so2, pm10, nh3, pm25_lag1, pm25_lag24
    ]])

    predicted_pm25 = model.predict(input_data)[0]

    # ----------------------------------
    # AQI Interpretation
    # ----------------------------------
    if predicted_pm25 <= 60:
        level = "Good"
        advice = "Air quality is safe. You can enjoy outdoor activities."
        color = "#1b5e20"
        bg = "#e8f5e9"

    elif predicted_pm25 <= 120:
        level = "Moderate"
        advice = "Air quality is acceptable. Sensitive people should be cautious."
        color = "#f9a825"
        bg = "#fffde7"

    elif predicted_pm25 <= 250:
        level = "Poor"
        advice = "Air quality is poor. Avoid prolonged outdoor exposure."
        color = "#ef6c00"
        bg = "#fff3e0"

    else:
        level = "Very Poor / Severe"
        advice = "Air quality is dangerous. Stay indoors and wear a mask if necessary."
        color = "#b71c1c"
        bg = "#ffebee"

    # ----------------------------------
    # Background Change
    # ----------------------------------
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {bg};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # ----------------------------------
    # Result Display
    # ----------------------------------
    st.subheader("📊 Prediction Result")

    st.markdown(
        f"""
        <div style="padding:20px; border-radius:10px; background-color:{color}; color:white;">
        <h3>Predicted PM2.5: {predicted_pm25:.1f}</h3>
        <h4>AQI Level: {level}</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(advice)

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.caption(
    "Built using Random Forest Regression on Delhi air pollution data."
)
