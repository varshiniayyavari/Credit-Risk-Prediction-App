import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model, scaler, feature names
with open("credit_model.pkl", "rb") as f:
    model = pickle.load(f)

scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))

st.title("💳 Credit Risk Prediction App")

st.write("Enter customer details below:")

# User Inputs (Main numeric features)
person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
person_income = st.number_input("Annual Income", min_value=0, value=50000)
person_emp_length = st.number_input("Employment Length (years)", min_value=0, value=2)
loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000)

if st.button("Predict"):

    # Create dictionary with all 22 features set to 0
    input_dict = dict.fromkeys(feature_names, 0)

    # Fill user-entered values
    input_dict["person_age"] = person_age
    input_dict["person_income"] = person_income
    input_dict["person_emp_length"] = person_emp_length
    input_dict["loan_amnt"] = loan_amnt

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠ High Risk of Default")
    else:
        st.success("✅ Low Risk - Safe Customer")
