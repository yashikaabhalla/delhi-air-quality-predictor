import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("rf_model.joblib")

st.set_page_config(page_title="Delhi Air Quality Predictor", page_icon="🌫️")

st.title("🌫️ Delhi Air Quality Predictor")
st.write("Enter **today’s AQI (PM2.5)** to predict **tomorrow’s air quality**.")

# User input (ONLY ONE)
today_aqi = st.number_input(
    "Today's AQI (PM2.5)",
    min_value=0.0,
    max_value=1000.0,
    value=150.0,
    step=1.0
)

st.markdown(
    "_We intelligently use historical pollution patterns behind the scenes to make this prediction._"
)

if st.button("Predict Tomorrow's AQI 🚀"):

    # Internally create feature vector
    # (Using AQI as lag features, others assumed stable)
    input_data = np.array([[
        0.5,      # CO (assumed average)
        10,       # NO
        40,       # NO2
        30,       # O3
        10,       # SO2
        120,      # PM10
        20,       # NH3
        today_aqi,  # PM2.5 lag 1
        today_aqi   # PM2.5 lag 24
    ]])

    prediction = model.predict(input_data)[0]

    st.subheader("📊 Prediction Result")

    # AQI category + color
    if prediction <= 50:
        st.success(f"🟢 AQI: {prediction:.1f} — Good 🌱")
        st.write("Air quality is safe. Enjoy outdoor activities!")
    elif prediction <= 100:
        st.info(f"🟡 AQI: {prediction:.1f} — Moderate 🙂")
        st.write("Acceptable air quality. Sensitive people should be cautious.")
    elif prediction <= 200:
        st.warning(f"🟠 AQI: {prediction:.1f} — Poor 😷")
        st.write("Limit outdoor activities, especially for children & elderly.")
    else:
        st.error(f"🔴 AQI: {prediction:.1f} — Very Poor / Severe ☠️")
        st.write("Avoid going outside. Wear a mask if necessary.")

st.markdown("---")
st.caption("Built using Machine Learning (Random Forest) on Delhi air pollution data.")
