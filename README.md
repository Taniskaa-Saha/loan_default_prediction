# Loan Default Prediction

## Project Overview

This project develops a Machine Learning model for predicting loan defaults using borrower and loan-related information. The goal is to identify applicants who are more likely to default, helping financial institutions improve credit risk assessment and lending decisions.

The project uses Random Forest Classification with SMOTE-based class balancing and probability threshold tuning to improve the detection of high-risk borrowers.

## Dataset

The dataset contains approximately 255,000 loan records with borrower, financial, and loan-related features.

### Target Variable

- **0 — No Default**
- **1 — Default**

### Key Features

- Age
- Income
- Loan Amount
- Credit Score
- Months Employed
- Number of Credit Lines
- Interest Rate
- Loan Term
- DTI Ratio
- Education
- Employment Type
- Marital Status
- Has Mortgage
- Has Dependents
- Loan Purpose
- Has Co-Signer

The dataset contains significantly fewer default cases than non-default cases, making class imbalance an important part of the modeling process.

## Technologies & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Random Forest
- SMOTE
- Streamlit

## Project Workflow

1. Dataset
2. Data Exploration
3. Data Preprocessing
4. Categorical Encoding
5. Train-Test Split
6. SMOTE Class Balancing
7. Random Forest Model
8. Probability Prediction
9. Threshold Tuning
10. Model Evaluation
11. Streamlit Application

## Class Imbalance

The dataset contains significantly fewer default cases than non-default cases. SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training data to balance the target classes.

### After SMOTE

- **Non-Default:** 180,524
- **Default:** 180,524

## Model

A Random Forest Classifier was selected for the prediction task because it performs well on structured/tabular data and can capture nonlinear relationships between features.

## Threshold Tuning

Instead of using the default probability threshold of 0.50, multiple thresholds were evaluated to improve detection of default cases.

The selected threshold was:

**0.27**

This resulted in a **0.53 recall for the default class**, while maintaining **0.74 overall accuracy**.

## Model Performance

| Metric | Result |
|---|---:|
| Accuracy | 74% |
| Default Recall | 53% |
| Default F1-Score | 32% |
| Selected Threshold | 0.27 |

The threshold was adjusted to place greater emphasis on identifying potential default cases rather than relying solely on overall accuracy.

## Live Demo

Try the deployed Loan Default Prediction application:

[Launch Streamlit App](https://loan-default-prediction-new.streamlit.app/)

## Project Structure

The repository contains the project files required for the machine learning workflow, model implementation, and Streamlit application.

## Key Result

The threshold tuning approach was used to improve the model's ability to identify potential loan defaults compared with relying solely on the default 0.50 classification threshold.

Because loan default prediction is an imbalanced classification problem, default-class recall and F1-score were considered alongside overall accuracy.

## Business Use Case

The project demonstrates how machine learning can support credit-risk assessment by identifying borrowers who may be at higher risk of loan default.