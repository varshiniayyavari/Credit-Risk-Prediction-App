import streamlit as st
import pickle
import pandas as pd

# Load trained pipeline
model = pickle.load(open("credit_model.pkl", "rb"))

st.title("💳 Credit Risk Prediction App (Professional Model)")

st.write("Enter customer details:")

# Numeric Inputs
person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
person_income = st.number_input("Annual Income", min_value=0, value=50000)
person_emp_length = st.number_input("Employment Length", min_value=0, value=2)
loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000)
loan_int_rate = st.number_input("Interest Rate", min_value=0.0, value=10.0)
loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, value=0.2)
cb_person_cred_hist_length = st.number_input("Credit History Length", min_value=0, value=5)

# Categorical Inputs
home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
cb_person_default_on_file = st.selectbox("Default History", ["Y", "N"])

if st.button("Predict"):

    input_df = pd.DataFrame([{
        "person_age": person_age,
        "person_income": person_income,
        "person_emp_length": person_emp_length,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "home_ownership": home_ownership,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "cb_person_default_on_file": cb_person_default_on_file
    }])

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("⚠ High Risk of Default")
    else:
        st.success("✅ Low Risk - Safe Customer")
