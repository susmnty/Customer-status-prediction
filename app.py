import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("best_model.pkl")

st.title("Customer Status Prediction")
st.write("Enter the customer details to predict their current status:")

# Categorical input mappings
categorical_options = {
    "Gender": ["Male", "Female"],
    "Education": ["Illiterate", "School", "Graduate", "Post-Graduate", "PhD"],
    "Occupation": ["Unemployed", "Labor", "Clerk", "Business", "Professional"],
    "Account type": ["Savings", "Current", "Salary", "Joint"],
    "Have minimum balance?": ["No", "Yes"],
    "Have multiple accounts?": ["No", "Yes"],
    "Own an active loan?": ["No", "Yes"],
    "Use internet or mobile banking?": ["No", "Yes"],
    "Has an active credit Card?": ["No", "Yes"],
    "Ever defaulted on a loan?": ["No", "Yes"],
    "Satisfied with bank service?": ["No", "Yes"],
    "Any transaction in the past 24 months": ["No", "Yes"]
}

# UI inputs for categorical features
cat_inputs = []
for feature, options in categorical_options.items():
    choice = st.selectbox(feature, options)
    cat_inputs.append(options.index(choice))

# UI inputs for numerical features
age = st.slider("Age", 18, 100)
opening_balance = st.number_input("Opening balance", min_value=0.0)
current_balance = st.number_input("Current balance", min_value=0.0)
distance = st.number_input("Distance of residence from the bank (km)", min_value=0.0)
activity_rate = st.slider("Quarterly activity rate", 0.0, 1.0)
months_since_txn = st.number_input("Months since last transaction", min_value=0)

num_inputs = [age, opening_balance, current_balance, distance, activity_rate, months_since_txn]

# Combine all features in correct order
final_input = np.array([cat_inputs + num_inputs])

# Prediction
if st.button("Predict"):
    prediction = model.predict(final_input)
    probabilities = model.predict_proba(final_input) if hasattr(model, 'predict_proba') else None

    st.success(f"Predicted Status: {prediction[0]}")
    if probabilities is not None:
        st.write(f"Prediction Confidence: {np.max(probabilities)*100:.2f}%")