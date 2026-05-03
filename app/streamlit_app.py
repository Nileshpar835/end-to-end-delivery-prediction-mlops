import streamlit as st
import requests

st.title("🚀 Delivery Time Predictor")
st.image("app/image1.png", use_container_width=True)
# Inputs
distance = st.number_input("Distance (km)", min_value=0.0)
prep_time = st.number_input("Preparation Time (min)", min_value=0)
experience = st.number_input("Courier Experience (years)", min_value=0.0)

weather = st.selectbox("Weather", ["Clear", "Rainy", "Foggy", "Windy"])
traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
vehicle = st.selectbox("Vehicle Type", ["Bike", "Scooter"])

if st.button("Predict"):

    data = {
        "Distance_km": distance,
        "Preparation_Time_min": prep_time,
        "Courier_Experience_yrs": experience,
        "Weather": weather,
        "Traffic_Level": traffic,
        "Time_of_Day": time_of_day,
        "Vehicle_Type": vehicle
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=data)
        result = response.json()

        st.success(f"Predicted Delivery Time: {result['prediction']} minutes")

    except Exception as e:
        st.error("API not running or error occurred")