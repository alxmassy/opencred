import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import pickle

# --------- DATA PROCESSING ----------

# Step 1: Load the dataset

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

print("Original dataset head:")
print(df.head())
print("\n Class Distribution (1 = Good, 2 = Bad):")
print(df['credit_risk'].value_counts())

# Step 2: Feature Engineering and Cleaning

df['credit_risk'] = df['credit_risk'].map({1: 0, 2: 1})  # Map to binary classification

X = df.drop('credit_risk', axis=1)
y = df['credit_risk']

numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

print(F"\nNumerical features: {numerical_features}")
print(F"\nCategorical features: {categorical_features}")

# Step 3: Create a preprocessing pipeline

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

# Step 5: Apply Preprocessing 

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Step 6: Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed,y_train)

print("\nClass distribution after SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

# --------- MODEL BUILDING ----------

# Step 1: Baseline Model: Logistic Regression
print("\n Training Baseline Model: Logistic Regression")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_resampled, y_train_resampled)

y_pred_lr = lr_model.predict(X_test_processed)
print(classification_report(y_test, y_pred_lr))
print(f"Logistic Regression ROC AUC: {roc_auc_score(y_test, lr_model.predict_proba(X_test_processed)[:, 1]):.4f}")

# Step 2: XGBoost Model
print("\n Training XGBoost Model")
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_resampled, y_train_resampled)

y_pred_xgb = xgb_model.predict(X_test_processed)
print("\nXGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))
print(f"XGBoost ROC AUC: {roc_auc_score(y_test, xgb_model.predict_proba(X_test_processed)[:, 1]):.4f}")

# Step 3: Save the Models

with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

print("\nXGBoost model saved to 'xgb_model.pkl'")
print("\nPreprocessor saved to 'preprocessor.pkl'")