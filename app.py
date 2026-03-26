import streamlit as st
import pandas as pd
import joblib

# load model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Loan Prediction App")

# INPUTS
age = st.number_input("Age")
income = st.number_input("Income")
loan_amount = st.number_input("Loan Amount")
credit_score = st.number_input("Credit Score")
num_transactions = st.number_input("Transactions")
annual_spend = st.number_input("Annual Spend")

city = st.selectbox("City", ["Bangalore","Chennai","Delhi","Hyderabad","Mumbai"])
employment = st.selectbox("Employment", ["Salaried","Self-Employed","Student","Unemployed"])
loan_type = st.selectbox("Loan Type", ["Home","Personal","Auto"])

# PREDICT
if st.button("Predict"):
    data = pd.DataFrame({
        'age':[age],
        'income':[income],
        'loan_amount':[loan_amount],
        'credit_score':[credit_score],
        'num_transactions':[num_transactions],
        'annual_spend':[annual_spend],
        'city':[city],
        'employment_type':[employment],
        'loan_type':[loan_type]
    })

    # encoding
    data = pd.get_dummies(data)

    # match columns
    data = data.reindex(columns=model_columns, fill_value=0)

    # scale
    data = scaler.transform(data)

    # predict
    result = model.predict(data)

    st.success(f"Prediction: {result[0]}")