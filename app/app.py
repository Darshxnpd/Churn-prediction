import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load("C:/Code/Churn-Prediction/data/WA_Fn-UseC_-Telco-Customer-Churn.pkl")
model_columns = joblib.load("C:/Code/Churn-Prediction/data/model_columns.pkl")

st.title("📉 Telco Churn Prediction App")

# User input form
st.sidebar.header("Enter Customer Details")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 10)
monthly = st.sidebar.slider("Monthly Charges", 10, 120, 70)
total = st.sidebar.slider("Total Charges", 0, 10000, 3000)
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Build input DataFrame
data = {
    'gender': gender,
    'SeniorCitizen': senior,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'MonthlyCharges': monthly,
    'TotalCharges': total,
    'Contract': contract,
    'InternetService': internet
}
# Build input DataFrame and encode using pd.get_dummies (same as training)
input_df = pd.DataFrame([data])
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# ...existing code...
if st.sidebar.button("Predict Churn"):
    try:
        prediction = model.predict(input_df)
        result = "Churn ❌" if prediction[0] == 0 else "Likely to Churn ⚠️"
        st.subheader(f"🔍 Prediction: **{result}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")