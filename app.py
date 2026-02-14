import streamlit as st
import pickle
import numpy as np

# Load model, scaler, feature names
model = pickle.load(open("credit_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))

st.title("💳 Credit Risk Prediction App")

st.write("Enter customer details below:")

# User Inputs
person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
person_income = st.number_input("Annual Income", min_value=0, value=50000)
person_emp_length = st.number_input("Employment Length (years)", min_value=0, value=2)
loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000)
loan_int_rate = st.number_input("Interest Rate", min_value=0.0, value=10.0)
loan_percent_income = st.number_input("Loan % of Income", min_value=0.0, value=0.2)
cb_person_cred_hist_length = st.number_input("Credit History Length", min_value=0, value=5)

if st.button("Predict"):
    input_data = np.array([[person_age, person_income, person_emp_length,
                            loan_amnt, loan_int_rate,
                            loan_percent_income,
                            cb_person_cred_hist_length]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠ High Risk of Default")
    else:
        st.success("✅ Low Risk - Safe Customer")
