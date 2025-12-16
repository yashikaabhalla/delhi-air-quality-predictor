import streamlit as st
import numpy as np
import pickle

# Load trained model
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Delhi Air Quality Predictor")
st.write("Predict PM2.5 based on pollution data")

co = st.number_input("CO")
no = st.number_input("NO")
no2 = st.number_input("NO2")
o3 = st.number_input("O3")
so2 = st.number_input("SO2")
pm10 = st.number_input("PM10")
nh3 = st.number_input("NH3")
lag1 = st.number_input("PM2.5 (1 hour ago)")
lag24 = st.number_input("PM2.5 (24 hours ago)")

if st.button("Predict PM2.5"):
    input_data = np.array([[co, no, no2, o3, so2, pm10, nh3, lag1, lag24]])
    prediction = model.predict(input_data)
    st.success(f"Predicted PM2.5: {prediction[0]:.2f}")
