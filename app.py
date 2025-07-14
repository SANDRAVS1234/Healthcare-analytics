import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and pre-processing tools
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
le_sex = joblib.load("le_sex.pkl")
le_source = joblib.load("le_source.pkl")  # just for reverse lookup if needed

# App title
st.title("EHR Patient Classification App")
st.write("Predict whether a patient will be **in care** or **out care** based on lab results.")

# Input form
with st.form("ehr_form"):
    haem = st.number_input("HAEMATOCRIT", min_value=0.0, value=35.0)
    hemo = st.number_input("HAEMOGLOBINS", min_value=0.0, value=11.0)
    ery = st.number_input("ERYTHROCYTE", min_value=0.0, value=4.5)
    leu = st.number_input("LEUCOCYTE", min_value=0.0, value=6.0)
    throm = st.number_input("THROMBOCYTE", min_value=0.0, value=310.0)
    mch = st.number_input("MCH", min_value=0.0, value=25.0)
    mchc = st.number_input("MCHC", min_value=0.0, value=33.0)
    mcv = st.number_input("MCV", min_value=0.0, value=75.0)
    age = st.number_input("AGE", min_value=0, value=30)
    sex = st.selectbox("SEX", options=le_sex.classes_)

    submitted = st.form_submit_button("Predict")

    if submitted:
        # Prepare input
        sex_encoded = le_sex.transform([sex])[0]
        input_data = np.array([[haem, hemo, ery, leu, throm, mch, mchc, mcv, age, sex_encoded]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        prediction_label = le_source.inverse_transform([prediction])[0]

        # Display result
        st.success(f"The predicted patient condition is: **{prediction_label.upper()}**")
        import streamlit as st

st.title("Healthcare Classification App")

with st.form("patient_form"):
    age = st.number_input("Age", min_value=0)
    sex = st.selectbox("Sex", ["M", "F"])
    haemoglobin = st.number_input("Haemoglobins")
    leukocyte = st.number_input("Leukocyte Count")

    submitted = st.form_submit_button("Submit")

    if submitted:
        st.write("Processing your input...")
        # You'd typically call your ML model here
        # prediction = model.predict(...)

