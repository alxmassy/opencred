import pickle
import shap 
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the saved model

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

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocess the test data
X_test_processed = preprocessor.transform(X_test)


# Step 2: Create a SHAP explainer

print("Creating SHAP explainer...")
explainer = shap.Explainer(model)
shap_values = explainer.shap_values(X_test_processed)

feature_names = preprocessor.get_feature_names_out()

# Step 3: Global feature importance

print("Generating global feature importance plot...")
shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names, show=False)

plt.title("Global Feature Importance (SHAP Summary Plot)")
plt.tight_layout()
plt.savefig('shap_summary.png')
plt.show()
print("Global feature importance plot saved as 'shap_summary.png'.")

# Step 4: Local explanations for a specific instance

print("Generating local explanations for a specific instance...")

X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)

shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test_processed_df.iloc[0],
    feature_names=feature_names
), show=False)

plt.title("SHAP Waterfall Plot for First Instance")
plt.tight_layout()
plt.savefig('shap_waterfall_plot.png')
plt.show()
print("Local explanation plot saved as 'shap_waterfall_plot.png'.")

print("SHAP analysis completed successfully.")