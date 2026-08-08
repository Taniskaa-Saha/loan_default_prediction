import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE #imbalanced dataset

df = pd.read_csv("Data/Loan_default.csv")
df.info()

# Check missing values
df.isnull().sum()

df.describe()

#show number of defaulters and non-defaulters
df['Default'].value_counts()

##VISUALIZATION
df['Default'].value_counts().plot(kind='bar')
plt.title('Loan Default Distribution')
plt.xlabel('Default (0 = No Default, 1 = Default)')
plt.ylabel('Number of Customers')
plt.show()

df.hist(figsize=(12, 10))
plt.show()

#Encode Categorical Features
df.dtypes
for col in df.select_dtypes(include='object').columns:
    print(col, df[col].unique())
df = df.drop("LoanID",axis=1)
df.columns

#One-Hot Encoding
df = pd.get_dummies(df, columns=[
    'Education',
    'EmploymentType',
    'MaritalStatus',
    'HasMortgage',
    'HasDependents',
    'LoanPurpose',
    'HasCoSigner'
], drop_first=True)
df.head()
df.dtypes

x = df.drop("Default",axis=1)
y = df["Default"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

#Applying SMOTE
smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

print("Before SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

#create RandomForest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train_smote, y_train_smote)
y_pred = rf_model.predict(x_test)
print(y_pred[:10])

#Model Evaluation
#generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.show()
print(classification_report(y_test, y_pred))