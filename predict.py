import streamlit as st
import pickle
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

# Load the pre-trained LightGBM model
with open('lightgbm_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataset
data = pd.read_csv('preprocessedfinal_mental_health_data_standardized.csv')

# Prepare the data
X = data.drop(columns=['Depression (%)'])
y = (data['Depression (%)'] > data['Depression (%)'].median()).astype(int)

# Feature Engineering
X['Schizophrenia_Bipolar'] = X['Schizophrenia (%)'] * X['Bipolar disorder (%)']
X['Anxiety_Drug'] = X['Anxiety disorders (%)'] * X['Drug use disorders (%)']

# Function to run LIME analysis
def run_lime_analysis():
    st.title("LIME Interpretability Analysis")

    # Select sample index for LIME analysis
    sample_idx = st.slider("Select Sample Index for LIME", 0, X.shape[0] - 1, 0)

    # Select the sample data
    input_data = X.iloc[[sample_idx]]

    # LIME interpretability
    explainer = LimeTabularExplainer(
        X.values,
        feature_names=X.columns,
        class_names=['Low', 'High'],
        mode='classification'
    )

    # Explain the selected instance
    explanation = explainer.explain_instance(input_data.iloc[0].values, model.predict_proba, num_features=5)
    st.write("Feature Importance for Selected Instance:")
    st.write(explanation.as_list())
    explanation.as_pyplot_figure()
    st.pyplot()
