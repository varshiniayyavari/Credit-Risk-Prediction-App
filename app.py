if st.button("Predict"):

    # Create empty dictionary with all 22 features
    input_dict = dict.fromkeys(feature_names, 0)

    # Fill the features user actually enters
    input_dict["person_age"] = person_age
    input_dict["person_income"] = person_income
    input_dict["person_emp_length"] = person_emp_length
    input_dict["loan_amnt"] = loan_amnt

    # Convert to DataFrame with correct column order
    input_df = pd.DataFrame([input_dict])

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠ High Risk of Default")
    else:
        st.success("✅ Low Risk - Safe Customer")
