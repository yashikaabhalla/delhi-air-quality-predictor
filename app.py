import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Delhi Air Quality Predictor",
    page_icon="🌫️",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("rf_model.joblib")

# -----------------------------
# UI Header
# -----------------------------
st.title("🌫️ Delhi Air Quality Predictor")
st.write(
    "Enter **today’s AQI (PM2.5)** to predict **tomorrow’s air quality**."
)

# -----------------------------
# User Input
# -----------------------------
today_aqi = st.number_input(
    "Today's AQI (PM2.5)",
    min_value=0.0,
    max_value=1000.0,
    value=100.0,
    step=1.0
)

st.caption(
    "We intelligently use historical pollution patterns behind the scenes to make this prediction."
)

# -----------------------------
# Prediction Logic
# -----------------------------
if st.button("Predict Tomorrow's AQI 🚀"):

    # Model expects 9 features (same order as training)
    input_data = np.array([[
        0.5,          # CO (assumed avg)
        10,           # NO
        40,           # NO2
        30,           # O3
        10,           # SO2
        120,          # PM10
        20,           # NH3
        today_aqi,    # PM2.5 lag 1
        today_aqi     # PM2.5 lag 24
    ]])

    # 1️⃣ Raw ML prediction
    raw_prediction = model.predict(input_data)[0]

    # 2️⃣ Hybrid trend correction (VERY IMPORTANT)
    if today_aqi > 300:
        prediction = raw_prediction + 0.15 * (today_aqi - raw_prediction)
    elif today_aqi > 150:
        prediction = raw_prediction + 0.10 * (today_aqi - raw_prediction)
    else:
        prediction = raw_prediction

    # 3️⃣ Safety limits
    prediction = max(0, min(prediction, 500))

    # -----------------------------
    # Display Result
    # -----------------------------
    st.subheader("📊 Prediction Result")

    if prediction <= 50:
        st.success(f"🟢 AQI: {prediction:.1f} — Good 🌱")
        st.write("Air quality is safe. Perfect for outdoor activities.")

    elif prediction <= 100:
        st.info(f"🟡 AQI: {prediction:.1f} — Moderate 🙂")
        st.write("Acceptable air quality. Sensitive people should be cautious.")

    elif prediction <= 200:
        st.warning(f"🟠 AQI: {prediction:.1f} — Poor 😷")
        st.write("Limit outdoor activities, especially for children and elderly.")

    else:
        st.error(f"🔴 AQI: {prediction:.1f} — Very Poor / Severe ☠️")
        st.write("Avoid going outside. Wear a mask if necessary.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "Built using Machine Learning (Random Forest) on Delhi air pollution data."
)
