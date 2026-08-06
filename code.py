import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
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