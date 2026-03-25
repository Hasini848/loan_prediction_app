import streamlit as st
import joblib
import pandas as pd

# load model
model = joblib.load("final_model.pkl")

st.title("Loan Prediction App")

# inputs
age = st.number_input("Age")
income = st.number_input("Income")

city = st.selectbox("City", ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Mumbai"])
employment = st.selectbox("Employment", ["Salaried", "Self-Employed", "Student", "Unemployed"])

# 👉 THIS IS IMPORTANT (input_data)
input_data = pd.DataFrame({
    'age': [age],
    'income': [income],
    'loan_amount': [0],
    'credit_score': [0],
    'num_transactions': [0],
    'annual_spend': [0],
    'day': [1],
    'month': [1],
    'year': [2024],

    'city_Bangalore': [1 if city == "Bangalore" else 0],
    'city_Chennai': [1 if city == "Chennai" else 0],
    'city_Delhi': [1 if city == "Delhi" else 0],
    'city_Hyderabad': [1 if city == "Hyderabad" else 0],
    'city_Mumbai': [1 if city == "Mumbai" else 0],

    'employment_type_Salaried': [1 if employment == "Salaried" else 0],
    'employment_type_Self-Employed': [1 if employment == "Self-Employed" else 0],
    'employment_type_Student': [1 if employment == "Student" else 0],
    'employment_type_Unemployed': [1 if employment == "Unemployed" else 0],

    'loan_type_Auto': [0],
    'loan_type_Education': [0],
    'loan_type_Home': [0],
    'loan_type_Personal': [0]
})

# predict
if st.button("Predict"):
    result = model.predict(input_data)
    st.success(f"Prediction: {result[0]}")