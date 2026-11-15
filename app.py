import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler (ensure pkl files are in the same directory)
model = joblib.load('cardio_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Cardiovascular Risk Predictor")
st.title("Cardiovascular Disease Risk Predictor")
st.markdown("Enter patient details to estimate CVD risk.")

# Input components
age = st.slider("Age (years)", 20, 90, 50)
gender = st.radio("Gender", [1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
height = st.slider("Height (cm)", 100, 220, 170)
weight = st.slider("Weight (kg)", 30, 200, 70)
ap_hi = st.slider("Systolic BP", 80, 250, 120)
ap_lo = st.slider("Diastolic BP", 40, 150, 80)
cholesterol = st.radio("Cholesterol", [1, 2, 3], format_func=lambda x: {1:"Normal",2:"Above Normal",3:"Well Above Normal"}[x])
gluc = st.radio("Glucose", [1, 2, 3], format_func=lambda x: {1:"Normal",2:"Above Normal",3:"Well Above Normal"}[x])
smoke = st.radio("Smoking", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
alco = st.radio("Alcohol", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
active = st.radio("Physically Active", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

if st.button("Predict"):
    bmi = weight / ((height / 100) ** 2)
    features = np.array([[age, gender, height, weight, ap_hi, ap_lo,
                          cholesterol, gluc, smoke, alco, active, bmi]])
    prob = model.predict_proba(scaler.transform(features))[0][1]
    st.success(f"Cardiovascular disease risk probability: **{prob:.2%}**")
