import streamlit as st
import pandas as pd
import joblib


# Load Model Artifacts
model = joblib.load("model/loan_default_model.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")
threshold = joblib.load("model/threshold.pkl")


# Page Configuration
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide"
)


# Header
st.title("🏦 Loan Default Prediction")

st.markdown(
    "Predict the likelihood of loan default using a machine learning model "
    "trained with Random Forest."
)

st.divider()


# Applicant Information
st.subheader("👤 Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

with col2:
    income = st.number_input(
        "Income",
        min_value=0,
        value=50000
    )

with col3:
    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650
    )


# Loan Information
st.subheader("💰 Loan Information")

col1, col2, col3 = st.columns(3)

with col1:
    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0,
        value=100000
    )

with col2:
    interest_rate = st.number_input(
        "Interest Rate (%)",
        min_value=0.0,
        value=10.0
    )

with col3:
    loan_term = st.number_input(
        "Loan Term (months)",
        min_value=1,
        value=36
    )


# Financial Information
st.subheader("📊 Financial Information")

col1, col2, col3 = st.columns(3)

with col1:
    months_employed = st.number_input(
        "Months Employed",
        min_value=0,
        value=60
    )

with col2:
    num_credit_lines = st.number_input(
        "Number of Credit Lines",
        min_value=0,
        value=5
    )

with col3:
    dti_ratio = st.number_input(
        "DTI Ratio",
        min_value=0.0,
        value=0.3
    )


# Personal & Loan Details
st.subheader("📋 Personal & Loan Details")

col1, col2, col3, col4 = st.columns(4)

with col1:
    education = st.selectbox(
        "Education",
        ["High School", "Bachelor's", "Master's", "PhD"]
    )

with col2:
    employment_type = st.selectbox(
        "Employment Type",
        ["Full-time", "Part-time", "Self-employed", "Unemployed"]
    )

with col3:
    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

with col4:
    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Home", "Auto", "Education", "Business", "Other"]
    )

col1, col2, col3 = st.columns(3)

with col1:
    has_mortgage = st.selectbox(
        "Has Mortgage",
        ["Yes", "No"]
    )

with col2:
    has_dependents = st.selectbox(
        "Has Dependents",
        ["Yes", "No"]
    )

with col3:
    has_cosigner = st.selectbox(
        "Has Co-Signer",
        ["Yes", "No"]
    )

st.divider()


# Prediction
if st.button("🔍 Predict Loan Default", use_container_width=True):

    applicant = pd.DataFrame({
        "Age": [age],
        "Income": [income],
        "LoanAmount": [loan_amount],
        "CreditScore": [credit_score],
        "MonthsEmployed": [months_employed],
        "NumCreditLines": [num_credit_lines],
        "InterestRate": [interest_rate],
        "LoanTerm": [loan_term],
        "DTIRatio": [dti_ratio],
        "Education": [education],
        "EmploymentType": [employment_type],
        "MaritalStatus": [marital_status],
        "HasMortgage": [has_mortgage],
        "HasDependents": [has_dependents],
        "LoanPurpose": [loan_purpose],
        "HasCoSigner": [has_cosigner]
    })

    # One-hot encoding
    applicant_encoded = pd.get_dummies(applicant)

    # Match training features
    applicant_encoded = applicant_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Predict probability
    probability = model.predict_proba(
        applicant_encoded
    )[0][1]

    # Apply saved threshold
    prediction = int(probability >= threshold)


    # Prediction Result
    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Default Probability",
            f"{probability:.2%}"
        )

    with result_col2:
        st.metric(
            "Decision Threshold",
            f"{threshold:.2f}"
        )

    if prediction == 1:

        st.error(
            "⚠️ High Risk — The applicant is predicted to default."
        )

    else:

        st.success(
            "✅ Low Risk — The applicant is predicted not to default."
        )

    st.caption(
        "The prediction is generated using a Random Forest model "
        "with a probability threshold of 0.27."
    )