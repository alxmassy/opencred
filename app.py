import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# --- 1. Load Artifacts ---
# Use Streamlit's caching to load the model and preprocessor only once.
@st.cache_data
def load_artifacts():
    with open('models/xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    return model, preprocessor

model, preprocessor = load_artifacts()

@st.cache_data
def load_explainer(_model, _preprocessor):
    # The explainer needs a background dataset for reference, here we use a dummy one
    # matching the structure after preprocessing.
    # Note: For more accurate SHAP values, you'd use a sample of your training data.
    feature_names = _preprocessor.get_feature_names_out()
    dummy_data = pd.DataFrame(columns=feature_names, data=[[0]*len(feature_names)])
    return shap.TreeExplainer(_model, dummy_data)

explainer = load_explainer(model, preprocessor)

# --- 2. App Title and Description ---
st.title("Creditworthiness Prediction App")
st.write(
    "This app predicts a person's creditworthiness (Good/Bad) based on their demographic "
    "and financial information. It also provides an explanation for each prediction."
)

# --- 3. User Input Section ---
st.sidebar.header("Applicant Information")

def get_user_input():
    # Use the exact categories from the dataset
    checking_account_map = {'< 0 DM': 'A11', '0 <= ... < 200 DM': 'A12', '>= 200 DM': 'A13', 'no checking account': 'A14'}
    credit_history_map = {'no credits/all paid': 'A30', 'all paid': 'A31', 'existing paid': 'A32', 'delay in paying': 'A33', 'critical account': 'A34'}
    purpose_map = {'car (new)': 'A40', 'car (used)': 'A41', 'furniture/equipment': 'A42', 'radio/TV': 'A43', 'domestic appliances': 'A44', 'repairs': 'A45', 'education': 'A46', 'vacation': 'A47', 'retraining': 'A48', 'business': 'A49', 'other': 'A410'}

    existing_checking_account = st.sidebar.selectbox('Checking Account Status', list(checking_account_map.keys()))
    duration_in_month = st.sidebar.slider('Duration in Month', 1, 80, 24)
    credit_history = st.sidebar.selectbox('Credit History', list(credit_history_map.keys()))
    purpose = st.sidebar.selectbox('Purpose', list(purpose_map.keys()))
    credit_amount = st.sidebar.number_input('Credit Amount', min_value=0, value=2500)
    age_in_years = st.sidebar.slider('Age in Years', 18, 75, 35)

    # Simplified inputs for other features
    savings_account_bonds = st.sidebar.selectbox('Savings Account/Bonds', ['A61', 'A62', 'A63', 'A64', 'A65'])
    present_employment_since = st.sidebar.selectbox('Present Employment Since', ['A71', 'A72', 'A73', 'A74', 'A75'])
    personal_status_sex = st.sidebar.selectbox('Personal Status and Sex', ['A91', 'A92', 'A93', 'A94'])
    property_type = st.sidebar.selectbox('Property', ['A121', 'A122', 'A123', 'A124'])
    housing = st.sidebar.selectbox('Housing', ['A151', 'A152', 'A153'])
    job = st.sidebar.selectbox('Job', ['A171', 'A172', 'A173', 'A174'])

    # Collect data into a dictionary
    data = {
        'existing_checking_account': checking_account_map[existing_checking_account],
        'duration_in_month': duration_in_month,
        'credit_history': credit_history_map[credit_history],
        'purpose': purpose_map[purpose],
        'credit_amount': credit_amount,
        'savings_account_bonds': savings_account_bonds,
        'present_employment_since': present_employment_since,
        'installment_rate_percentage': 4,  # Default value
        'personal_status_sex': personal_status_sex,
        'other_debtors_guarantors': 'A101', # Default
        'present_residence_since': 4, # Default
        'property': property_type,
        'age_in_years': age_in_years,
        'other_installment_plans': 'A143', # Default
        'housing': housing,
        'number_of_existing_credits': 1, # Default
        'job': job,
        'number_of_people_liable': 1, # Default
        'telephone': 'A192', # Default (yes)
        'foreign_worker': 'A201' # Default (yes)
    }
    return pd.DataFrame(data, index=[0])

input_df = get_user_input()

# --- 4. Prediction and Explanation ---
if st.sidebar.button("Predict Creditworthiness"):
    # Preprocess the user input
    input_processed = preprocessor.transform(input_df)

    # Get prediction and probability
    prediction = model.predict(input_processed)[0]
    probability = model.predict_proba(input_processed)[0]

    st.subheader("Prediction Result")
    if prediction == 0:
        st.success("✅ Credit Approved (Good Credit)")
    else:
        st.error("❌ Credit Denied (Bad Credit)")

    st.write(f"**Confidence Score:**")
    st.write(f"Good Credit: **{probability[0]*100:.2f}%** | Bad Credit: **{probability[1]*100:.2f}%**")

    # --- SHAP Explanation ---
    st.subheader("Why was this decision made?")

    # Get feature names after preprocessing
    feature_names = preprocessor.get_feature_names_out()
    input_processed_df = pd.DataFrame(input_processed, columns=feature_names)

    # Calculate SHAP values for the single instance
    shap_values = explainer.shap_values(input_processed_df.iloc[0])
    
    # --- DEBUGGING: Inspect the SHAP values ---
    st.write("--- Debug Info ---")
    st.write("Shape of SHAP values:", shap_values.shape)
    st.write("SHAP values:", shap_values)
    st.write("--------------------")

    plt.clf()
    
    # Create the force plot and capture it
    force_plot = shap.force_plot(
        base_value=explainer.expected_value,
        shap_values=shap_values,
        features=input_processed_df.iloc[0],
        matplotlib=True,
        show=False,
        text_rotation=15
    )
    
    # Render the plot in Streamlit
    st.pyplot(force_plot, bbox_inches='tight')
    st.write(
        "The plot above shows the features that contributed to the final decision. "
        "Features in **red** pushed the prediction towards 'Bad Credit', while features in "
        "**blue** pushed it towards 'Good Credit'."
    )

else:
    st.info("Please fill in the applicant's details in the sidebar and click 'Predict'.")