from fastapi import FastAPI
import joblib
import pandas as pd

# Create FastAPI application
app = FastAPI(title="Loan Default Prediction API")

# Load saved model artifacts
model = joblib.load("model/loan_default_model.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")
threshold = joblib.load("model/threshold.pkl")


@app.get("/")
def home():
    return {"message": "Loan Default Prediction API is running"}


@app.post("/predict")
def predict(
    Age: int,
    Income: int,
    LoanAmount: int,
    CreditScore: int,
    MonthsEmployed: int,
    NumCreditLines: int,
    InterestRate: float,
    LoanTerm: int,
    DTIRatio: float,
    Education: str,
    EmploymentType: str,
    MaritalStatus: str,
    HasMortgage: str,
    HasDependents: str,
    LoanPurpose: str,
    HasCoSigner: str
):

    # Create applicant DataFrame
    applicant = {
        "Age": Age,
        "Income": Income,
        "LoanAmount": LoanAmount,
        "CreditScore": CreditScore,
        "MonthsEmployed": MonthsEmployed,
        "NumCreditLines": NumCreditLines,
        "InterestRate": InterestRate,
        "LoanTerm": LoanTerm,
        "DTIRatio": DTIRatio,
        "Education": Education,
        "EmploymentType": EmploymentType,
        "MaritalStatus": MaritalStatus,
        "HasMortgage": HasMortgage,
        "HasDependents": HasDependents,
        "LoanPurpose": LoanPurpose,
        "HasCoSigner": HasCoSigner
    }

    applicant_df = pd.DataFrame([applicant])

    # One-hot encoding
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

    # Match training features
    applicant_df = applicant_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Predict probability
    probability = model.predict_proba(applicant_df)[0, 1]

    # Apply threshold
    prediction = int(probability >= threshold)

    return {
        "default_probability": round(float(probability), 4),
        "prediction": prediction,
        "result": (
            "Likely to Default"
            if prediction == 1
            else "Not Likely to Default"
        )
    }