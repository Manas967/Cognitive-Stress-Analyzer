import os
import pickle
import streamlit as st #type:ignore
import pandas as pd #type:ignore
from services.gemini_engine import generate_cognitive_report
import services.model as model

model_dir = "models"
model_path = os.path.join(model_dir, "stress_model.pkl")
scaler_path = os.path.join(model_dir, "scaler.pkl")

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    model.train_and_save_pipeline()

with open(model_path, "rb") as m_file:
    ml_model = pickle.load(m_file)
with open(scaler_path, "rb") as s_file:
    scaler = pickle.load(s_file)

st.set_page_config(page_title="Cognitive Stress Analyzer", layout="wide")
st.title("🧠 Cognitive Stress Analyzer")
st.write("MCA-3 Project Submission by Manas Tiwari and KM Ankita Jaiswal (Group-15)")

st.header("1. Objective Biological Inputs")
pulse_count = st.number_input("Enter 15-second pulse count:", min_value=10, max_value=40, value=18)
calculated_bpm = pulse_count * 4

st.info(f"Calculated Heart Rate: {calculated_bpm} BPM")
sleep_hours = st.slider("Hours of Sleep Last Night:", 0.0, 12.0, 6.5, step=0.5)

st.header("2. Subjective & Environmental Context")
work_hours = st.slider("Continuous Work Duration (Hours):", 1, 12, 4)

noise_level = st.selectbox("Environment Noise Level:", options=[1, 2, 3], format_func=lambda x: ["Low/Quiet", "Medium", "High/Noisy"][x-1])

user_text_context = st.text_area("How are you feeling right now?", "Feeling rushed.")

if st.button("Analyze Stress Level"):
    input_data = pd.DataFrame({
        'heart_rate': [calculated_bpm],
        'work_duration_hours': [work_hours],
        'environment_noise': [noise_level],
        'sleep_hours': [sleep_hours]
    })
    
    features_scaled = scaler.transform(input_data)
    prediction = ml_model.predict(features_scaled)[0]
    
    stress_labels = {0: "Neutral / Low Stress", 1: "Time Pressure / Moderate Stress", 2: "High Stress"}
    result_text = stress_labels[prediction]
    
    st.subheader(f"Machine Learning Result: **{result_text}**")
    st.write("---")
    st.subheader("🤖 Gemini Explainable AI Diagnostics")
    
    with st.spinner("Gemini is analyzing the cognitive loop..."):
        report = generate_cognitive_report(
            calculated_bpm, 
            sleep_hours, 
            work_hours, 
            noise_level, 
            user_text_context, 
            result_text
        )
        st.write(report)