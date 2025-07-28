import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference, selection_rate


# Step 1: Load Models, Preprocessor and Data

print("Loading the model, preprocessor and data...")

with open('models/xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)  

with open('models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Load the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
column_names = [
    'existing_checking_account', 'duration_in_month', 'credit_history', 'purpose',
    'credit_amount', 'savings_account_bonds', 'present_employment_since',
    'installment_rate_percentage', 'personal_status_sex', 'other_debtors_guarantors',
    'present_residence_since', 'property', 'age_in_years', 'other_installment_plans',
    'housing', 'number_of_existing_credits', 'job', 'number_of_people_liable',
    'telephone', 'foreign_worker', 'credit_risk'
]
df = pd.read_csv(url, sep=' ', header=None, names=column_names)
df['credit_risk'] = df['credit_risk'].map({1: 0, 2: 1})
X = df.drop('credit_risk', axis=1)
y = df['credit_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 2: Make predictions on Test Data
X_test_processed = preprocessor.transform(X_test)
y_pred = model.predict(X_test_processed)

# Step 3: Define Sensitive Features
sensitive_features_sex = X_test['personal_status_sex'].apply(lambda x: 'male' if x in ['A91', 'A93', 'A94'] else 'female')
sensitive_features_age = X_test['age_in_years'].apply(lambda x: 'age_<25' if x <=25 else 'age_>25')

# Step 4: Calculate Fairness Metrics

metrics = {
    'selection_rate': lambda y_true, y_pred: selection_rate(y_true, y_pred, pos_label=0)
}

grouped_on_sex = MetricFrame(metrics=metrics,
                             y_true=y_test,
                             y_pred=y_pred,
                             sensitive_features=sensitive_features_sex)

grouped_on_age = MetricFrame(metrics=metrics,
                             y_true=y_test,
                             y_pred=y_pred,
                             sensitive_features=sensitive_features_age)

print("\n--- Fairness Audit Report ---")
print("\n--- Based on Sex ---")
print("Selection Rate by Group (proportion approved for credit):")
print(grouped_on_sex.by_group)
print(f"\nDemographic Parity Difference: {demographic_parity_difference(y_true=y_test, y_pred=(y_pred==0), sensitive_features=sensitive_features_sex):.4f}")
print(f"Equalized Odds Difference: {equalized_odds_difference(y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_features_sex):.4f}")


print("\n--- Based on Age ---")
print("Selection Rate by Group (proportion approved for credit):")
print(grouped_on_age.by_group)
print(f"\nDemographic Parity Difference: {demographic_parity_difference(y_true=y_test, y_pred=(y_pred==0), sensitive_features=sensitive_features_age):.4f}")
print(f"Equalized Odds Difference: {equalized_odds_difference(y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_features_age):.4f}")

print("\nNote: A difference value closer to 0 is considered more fair.")