import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# Load model
model = pickle.load(open("credit_model.pkl", "rb"))

# Load dataset for dashboard analysis
df = pd.read_csv("credit_risk_dataset.csv")

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("💳 Credit Risk Dashboard")
page = st.sidebar.radio("Navigate", 
                        ["🏠 Prediction", 
                         "📊 Data Insights", 
                         "📉 Model Info",
                         "📂 Dataset Explorer"])

# ---------------- PAGE 1 ---------------- #
if page == "🏠 Prediction":

    st.title("Credit Risk Prediction")

    col1, col2 = st.columns(2)

    with col1:
        person_age = st.number_input("Age", 18, 100, 30)
        person_income = st.number_input("Annual Income", 0, 1000000, 50000)
        person_emp_length = st.number_input("Employment Length (years)", 0, 50, 5)
        loan_amnt = st.number_input("Loan Amount", 0, 1000000, 10000)
        loan_int_rate = st.number_input("Interest Rate", 0.0, 50.0, 10.0)

    with col2:
        loan_percent_income = st.number_input("Loan % of Income", 0.0, 1.0, 0.2)
        cb_person_cred_hist_length = st.number_input("Credit History Length", 0, 50, 5)
        person_home_ownership = st.selectbox("Home Ownership", df["person_home_ownership"].unique())
        loan_intent = st.selectbox("Loan Intent", df["loan_intent"].unique())
        loan_grade = st.selectbox("Loan Grade", df["loan_grade"].unique())
        cb_person_default_on_file = st.selectbox("Previous Default", df["cb_person_default_on_file"].unique())

    if st.button("Predict Risk"):

        input_df = pd.DataFrame([{
            "person_age": person_age,
            "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": person_emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": cb_person_cred_hist_length
        }])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(f"⚠ High Risk of Default\nProbability: {probability:.2%}")
        else:
            st.success(f"✅ Low Risk Customer\nProbability: {probability:.2%}")

# ---------------- PAGE 2 ---------------- #
elif page == "📊 Data Insights":

    st.title("Data Insights")

    fig1 = px.histogram(df, x="loan_amnt", title="Loan Amount Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(df, x="person_income", title="Income Distribution")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.box(df, x="loan_status", y="loan_amnt", title="Loan Amount vs Default")
    st.plotly_chart(fig3, use_container_width=True)

# ---------------- PAGE 3 ---------------- #
elif page == "📉 Model Info":

    st.title("Model Information")

    st.write("Model Type: Logistic Regression (Pipeline)")
    st.write("Preprocessing: ColumnTransformer + OneHotEncoding + Scaling")
    st.write("Features Used:")

    st.write(model.named_steps["preprocessor"].feature_names_in_)

# ---------------- PAGE 4 ---------------- #
elif page == "📂 Dataset Explorer":

    st.title("Dataset Explorer")

    income_filter = st.slider("Filter by Income", 
                              int(df["person_income"].min()), 
                              int(df["person_income"].max()), 
                              (0, int(df["person_income"].max())))

    filtered_df = df[(df["person_income"] >= income_filter[0]) & 
                     (df["person_income"] <= income_filter[1])]

    st.dataframe(filtered_df)

    st.download_button("Download Filtered Data", 
                       filtered_df.to_csv(index=False), 
                       "filtered_data.csv")
