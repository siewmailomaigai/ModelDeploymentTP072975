import streamlit as st
import pickle
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load the pre-trained model
with open('lightgbm_model.pkl', 'rb') as file:
    model = pickle.load(file)

def run_prediction_page():
    st.title("Mental Health Prediction with LightGBM")

    # Load and preprocess the dataset
    data = pd.read_csv('path_to_data/preprocessedfinal_mental_health_data_standardized.csv')

    # Drop non-numeric columns and target
    X = data.drop(columns=['index', 'Entity', 'Code', 'Year', 'Depression (%)'])
    y = (data['Depression (%)'] > data['Depression (%)'].median()).astype(int)

    # Feature Engineering
    X['Schizophrenia_Bipolar'] = X['Schizophrenia (%)'] * X['Bipolar disorder (%)']
    X['Anxiety_Drug'] = X['Anxiety disorders (%)'] * X['Drug use disorders (%)']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Apply SMOTE to balance the classes in the training set
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

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
        explainer = LimeTabularExplainer(X_train_smote.values, feature_names=X_train.columns, class_names=['Low', 'High'], mode='classification')
        explanation = explainer.explain_instance(input_data.iloc[0].values, model.predict_proba, num_features=5)
        st.write("Feature Importance for Prediction:")
        st.write(explanation.as_list())
        explanation.as_pyplot_figure()
        st.pyplot()

