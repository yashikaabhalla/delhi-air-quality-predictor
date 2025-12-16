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
    "Enter current pollution levels to predict **tomorrow’s PM2.5** and understand its **AQI impact**."
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

st.markdown("### ⏳ PM2.5 History")
pm25_lag1 = st.number_input("PM2.5 (1 hour ago)", 0.0, 1000.0, 100.0)
pm25_lag24 = st.number_input("PM2.5 (24 hours ago)", 0.0, 1000.0, 100.0)

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("Predict Air Quality 🚨"):

    input_data = np.array([[
        co, no, no2, o3, so2, pm10, nh3, pm25_lag1, pm25_lag24
    ]])

    predicted_pm25 = model.predict(input_data)[0]

    # ----------------------------------
    # AQI Mapping + UI Theme
    # ----------------------------------
    if predicted_pm25 <= 60:
        level = "Good"
        advice = "Air quality is safe. Outdoor activities are perfectly fine."
        bg_gradient = "linear-gradient(135deg, #1b5e20, #2e7d32)"

    elif predicted_pm25 <= 120:
        level = "Moderate"
        advice = "Air quality is acceptable. Sensitive individuals should be cautious."
        bg_gradient = "linear-gradient(135deg, #f9a825, #f57f17)"

    elif predicted_pm25 <= 250:
        level = "Poor"
        advice = "Air quality is poor. Reduce outdoor exposure."
        bg_gradient = "linear-gradient(135deg, #ef6c00, #e65100)"

    else:
        level = "Very Poor / Severe"
        advice = "Air quality is dangerous. Stay indoors and wear a mask if necessary."
        bg_gradient = "linear-gradient(135deg, #b71c1c, #7f0000)"

    # ----------------------------------
    # Background Styling (Dark + Glow)
    # ----------------------------------
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {bg_gradient};
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # ----------------------------------
    # Result Card
    # ----------------------------------
    st.subheader("📊 Prediction Result")

    st.markdown(
        f"""
        <div style="
            padding:25px;
            border-radius:15px;
            background: rgba(0,0,0,0.35);
            backdrop-filter: blur(10px);
            box-shadow: 0 0 40px rgba(0,0,0,0.4);
            color:white;
        ">
            <h2>Predicted PM2.5: {predicted_pm25:.1f}</h2>
            <h3>AQI Level: {level}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(f"### 🩺 Health Advisory\n{advice}")

    # ----------------------------------
    # Knowledge Section
    # ----------------------------------
    st.markdown("---")
    st.markdown("### 📘 How PM2.5 Affects AQI")

    st.markdown(
        """
        - **PM2.5** refers to fine particulate matter smaller than 2.5 microns.
        - These particles penetrate deep into the lungs and bloodstream.
        - **AQI (Air Quality Index)** is calculated using PM2.5 as a primary pollutant.
        - When **PM2.5 increases**, the **AQI value also increases**, indicating worse air quality.
        - Higher AQI levels are associated with **respiratory, cardiac, and health risks**.
        """
    )

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.caption(
    "Built using Random Forest Regression on real Delhi air pollution data."
)
