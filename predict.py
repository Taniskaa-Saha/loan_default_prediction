import joblib
import pandas as pd

# Load saved model artifacts
model = joblib.load("model/loan_default_model.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")
threshold = joblib.load("model/threshold.pkl")

# New loan applicant
new_applicant = {
    "Age": 35,
    "Income": 60000,
    "LoanAmount": 200000,
    "CreditScore": 720,
    "MonthsEmployed": 60,
    "NumCreditLines": 4,
    "InterestRate": 8.5,
    "LoanTerm": 36,
    "DTIRatio": 0.25,
    "Education": "Bachelor's",
    "EmploymentType": "Full-time",
    "MaritalStatus": "Married",
    "HasMortgage": "Yes",
    "HasDependents": "No",
    "LoanPurpose": "Home",
    "HasCoSigner": "Yes"
}

# Convert applicant information into DataFrame
applicant_df = pd.DataFrame([new_applicant])

# One-hot encode categorical data
applicant_df = pd.get_dummies(
    applicant_df,
    columns=[
        "Education",
        "EmploymentType",
        "MaritalStatus",
        "HasMortgage",
        "HasDependents",
        "LoanPurpose",
        "HasCoSigner"
    ],
    drop_first=True
)

# Make sure applicant has exactly the same features as the model
applicant_df = applicant_df.reindex(
    columns=feature_columns,
    fill_value=0
)

# Prediction
probability = model.predict_proba(applicant_df)[:, 1][0]

prediction = int(probability >= threshold)

print("Default Probability:", round(probability, 4))
print("Prediction:", prediction)

if prediction == 1:
    print("Result: HIGH RISK - Likely to Default")
else:
    print("Result: LOWER RISK - Not Likely to Default")