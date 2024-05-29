import pandas as pd
import streamlit as st
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Streamlit app
st.title("Heart Disease Prediction")
st.write("This app predicts heart disease using a Bayesian Network.")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    heart_disease = pd.DataFrame(data)
    st.write("### The first 5 values of data")
    st.write(heart_disease.head())

    # Define the Bayesian Model
    model = BayesianModel([
        ('age', 'Lifestyle'),
        ('Gender', 'Lifestyle'),
        ('Family', 'heartdisease'),
        ('diet', 'cholestrol'),
        ('Lifestyle', 'diet'),
        ('cholestrol', 'heartdisease'),
        ('diet', 'cholestrol')
    ])

    # Fit the model using Maximum Likelihood Estimation
    model.fit(heart_disease, estimator=MaximumLikelihoodEstimator)

    # Perform inference
    HeartDisease_infer = VariableElimination(model)

    st.write("### Enter the following information for prediction:")
    age = st.selectbox('Age', ['SuperSeniorCitizen', 'SeniorCitizen', 'MiddleAged', 'Youth', 'Teen'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    family_history = st.selectbox('Family History', ['Yes', 'No'])
    diet = st.selectbox('Diet', ['High', 'Medium'])
    lifestyle = st.selectbox('Lifestyle', ['Athlete', 'Active', 'Moderate', 'Sedentary'])
    cholesterol = st.selectbox('Cholesterol', ['High', 'BorderLine', 'Normal'])

    age_mapping = {'SuperSeniorCitizen': 0, 'SeniorCitizen': 1, 'MiddleAged': 2, 'Youth': 3, 'Teen': 4}
    gender_mapping = {'Male': 0, 'Female': 1}
    family_history_mapping = {'Yes': 1, 'No': 0}
    diet_mapping = {'High': 0, 'Medium': 1}
    lifestyle_mapping = {'Athlete': 0, 'Active': 1, 'Moderate': 2, 'Sedentary': 3}
    cholesterol_mapping = {'High': 0, 'BorderLine': 1, 'Normal': 2}

    if st.button("Predict"):
        q = HeartDisease_infer.query(variables=['heartdisease'], evidence={
            'age': age_mapping[age],
            'Gender': gender_mapping[gender],
            'Family': family_history_mapping[family_history],
            'diet': diet_mapping[diet],
            'Lifestyle': lifestyle_mapping[lifestyle],
            'cholestrol': cholesterol_mapping[cholesterol]
        })

        st.write("### Prediction")
        st.write(q['heartdisease'])
else:
    st.write("Please upload a CSV file.")
