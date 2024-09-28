import streamlit as st
import pickle
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import lightgbm as lgb
import numpy as np

# Load the pre-trained model
with open('lightgbm_model.pkl', 'rb') as file:
    model = pickle.load(file)

def run_prediction_page():
    st.title("Mental Health Prediction with LightGBM")

    # User inputs for model features
    schizophrenia = st.slider("Schizophrenia (%)", 0.0, 100.0, 10.0)
    bipolar = st.slider("Bipolar Disorder (%)", 0.0, 100.0, 10.0)
    anxiety = st.slider("Anxiety Disorders (%)", 0.0, 100.0, 10.0)
    drug_use = st.slider("Drug Use Disorders (%)", 0.0, 100.0, 10.0)
    alcohol_use = st.slider("Alcohol Use Disorders (%)", 0.0, 100.0, 10.0)

    # Create dataframe for input
    input_data = pd.DataFrame({
        'Schizophrenia (%)': [schizophrenia],
        'Bipolar disorder (%)': [bipolar],
        'Anxiety disorders (%)': [anxiety],
        'Drug use disorders (%)': [drug_use],
        'Alcohol use disorders (%)': [alcohol_use]
    })

    # Make prediction
    if st.button("Predict"):
        prediction = model.predict(input_data)
        st.write(f"Prediction: {'High Risk' if prediction[0] == 1 else 'Low Risk'}")

        # LIME Interpretability
        explainer = LimeTabularExplainer(input_data.values, feature_names=input_data.columns, class_names=['Low', 'High'], mode='classification')
        explanation = explainer.explain_instance(input_data.iloc[0].values, model.predict_proba, num_features=5)
        st.write("Feature Importance for Prediction:")
        st.write(explanation.as_list())
        explanation.as_pyplot_figure()
        st.pyplot()
