import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("loan_model.pkl")

# Page Config
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

# Title
st.title("🏦 Loan Approval Prediction")
st.markdown("Check whether your loan is likely to be approved.")

# Inputs
gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

married = st.selectbox(
    "Married",
    ["Yes", "No"]
)

income = st.number_input(
    "Applicant Income",
    min_value=0,
    value=5000,
    step=100
)

loan_amount_rupees = st.number_input(
    "Loan Amount (₹)",
    min_value=50000,
    max_value=5000000,
    value=200000,
    step=10000,
    format="%d"
)

dependents = st.selectbox(
    "Dependents",
    [0, 1, 2, 3]
)
credit_history = st.selectbox(
    "Credit History",
    ["Good (1)", "Bad (0)"]
)

credit_history = 1 if credit_history == "Good (1)" else 0

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["No", "Yes"]
)

coapplicant_income = st.number_input(
    "Coapplicant Income",
    min_value=0,
    value=0
)

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

# Convert loan amount
loan_amount = loan_amount_rupees / 1000

# Prediction
if st.button("Predict Loan Status"):

    data = np.array([
        1 if gender == "Male" else 0,
        1 if married == "Yes" else 0,
        dependents,
        1 if education == "Graduate" else 0,
        1 if self_employed == "Yes" else 0,
        income,
        coapplicant_income,
        loan_amount,
        360,
        credit_history,
        0 if property_area == "Rural"
        else 1 if property_area == "Semiurban"
        else 2
    ]).reshape(1, -1)

    prediction = model.predict(data)

    probability = model.predict_proba(data)

    approval_prob = probability[0][1] * 100
    rejection_prob = probability[0][0] * 100

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
        st.balloons()
    else:
        st.error("❌ Loan Rejected")

    st.write(f"📈 Approval Probability: {approval_prob:.2f}%")
    st.write(f"📉 Rejection Probability: {rejection_prob:.2f}%")