import streamlit as st
import pandas as pd
import pickle as pkl

import numpy as np
from keras.models import load_model

# Load the saved model
model = load_model("model.h5")

# Load the saved scaler and label encoder
with open('sc.pkl', 'rb') as file:
    scaler = pkl.load(file)

with open('lbl_encoder.pkl', 'rb') as file:
    label_encoder = pkl.load(file)

# Streamlit app
st.title("Customer Churn Prediction")

# Sidebar with user input
st.sidebar.header("User Input Features")

# Text input boxes for user input
total_charges = st.sidebar.text_input("Total Charges", "0.0")
monthly_charges = st.sidebar.text_input("Monthly Charges", "0.0")
tenure = st.sidebar.slider("Tenure", 0, 100, 50)

# Extend the label encoder for Contract and PaymentMethod
contract_options = ['Month-to-month', 'One year', 'Two year']
selected_contract = st.sidebar.selectbox("Contract", contract_options)

payment_method_options = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
selected_payment_method = st.sidebar.selectbox("Payment Method", payment_method_options)

# Extend the label encoder classes using numpy arrays
label_encoder.classes_ = np.concatenate((label_encoder.classes_, np.array(contract_options), np.array(payment_method_options)))

# Encode categorical features
selected_contract_encoded = label_encoder.transform([selected_contract])[0]
selected_payment_method_encoded = label_encoder.transform([selected_payment_method])[0]

# Create a DataFrame with user input
user_input = pd.DataFrame({
    'TotalCharges': [float(total_charges)],
    'MonthlyCharges': [float(monthly_charges)],
    'tenure': [tenure],
    'Contract': [selected_contract_encoded],
    'PaymentMethod': [selected_payment_method_encoded]
})

# Scale the input using the saved scaler
user_input_scaled = scaler.transform(user_input)

# Predict button
if st.sidebar.button("Predict"):
    # Make predictions
    probability_of_churning = model.predict(user_input_scaled)

    
    churn_threshold = 0.5  

    # Display prediction with churn probability
    st.subheader("Churning Prediction")
    st.write(f"The probability of the customer churning is: {probability_of_churning[0, 0]:.2f}")

    if probability_of_churning[0, 0] > churn_threshold:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is likely to stay.")
