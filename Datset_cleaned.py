# Bringing the dataset into the project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib as jwb
# Load the dataset into a pandas DataFrame
Loan_data=pd.read_csv('loan_approval.csv')
# Display the first few rows of the dataset
print(Loan_data.head())
# Check for missing values in the dataset
print(Loan_data.isnull().sum())
# Check the data types of the columns
print(Loan_data.dtypes)
# Summary statistics of the dataset
print(Loan_data.describe())
# Visualize the distribution of the target variable (Loan_Status)
sns.countplot(x='loan_amount', data=Loan_data)
plt.title('Distribution of Loan Amount')    
plt.xlabel('Loan Amount')
plt.ylabel('Count')
plt.show()
# Cleaning the dataset by dropping rows with missing values 
Loan_data_cleaned = Loan_data.dropna()
# Verify that there are no missing values in the cleaned dataset
print(Loan_data_cleaned.isnull().sum())
print(Loan_data_cleaned.head())
Loan_data_cleaned['loan_amount'].value_counts(normalize=True)
print(Loan_data_cleaned.head())
