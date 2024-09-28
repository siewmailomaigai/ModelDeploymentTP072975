import streamlit as st
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv('preprocessedfinal_mental_health_data_standardized.csv')

# Prepare the data
X = data.drop(columns=['Depression (%)'])
y = (data['Depression (%)'] > data['Depression (%)'].median()).astype(int)

# Feature Engineering
X['Schizophrenia_Bipolar'] = X['Schizophrenia (%)'] * X['Bipolar disorder (%)']
X['Anxiety_Drug'] = X['Anxiety disorders (%)'] * X['Drug use disorders (%)']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SMOTE to balance classes
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

def run_tuning_page():
    st.title("Tune LightGBM Model Parameters")

    # Hyperparameter sliders for tuning
    n_estimators = st.slider("Number of Boosting Rounds (n_estimators)", 50, 200, 100)
    learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1)
    max_depth = st.slider("Max Depth", -1, 20, -1)

    # Train model with tuned parameters
    model = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
    model.fit(X_train_smote, y_train_smote)

    # Evaluate model
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    accuracy = accuracy_score(y_test, y_pred)

    # Display metrics
    st.write(f"Test AUC: {auc:.4f}")
    st.write(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Save the model for LIME analysis
    import pickle
    with open('lightgbm_model.pkl', 'wb') as file:
        pickle.dump(model, file)
