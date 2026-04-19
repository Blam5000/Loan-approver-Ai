import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import XGBRegressor 
from sklearn.metrics import accuracy_score, mean_squared_error , r2_score , mean_absolute_error
from sklearn.model_selection import train_test_split
from Datset_goofy import Loan_data_cleaned
import joblib as jwb
# Define the features and target variable
X = Loan_data_cleaned[['loan_amount', 'credit_score', 'income']]
y = Loan_data_cleaned['loan_approved']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=42)
# Initialize the XGBoost Classifier
loan_approval_model = XGBClassifier(n_estimators=10000, learning_rate=0.01, max_depth=50, random_state=42, min_depth=25, min_leaf_nodes=10)
# Train the model on the training data
loan_approval_model.fit(X_train, y_train)
# Make predictions on the test set
loan_pred = loan_approval_model.predict(X_test)
# Evaluate the model's performance
accuracy = accuracy_score(y_test, loan_pred)
print(f'Accuracy: {accuracy:.2f}')
# Save the trained model using joblib
jwb.dump(loan_approval_model, 'loan_approval_model_xgb.pkl')
