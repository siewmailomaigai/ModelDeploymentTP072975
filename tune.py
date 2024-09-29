import streamlit as st
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pandas as pd
from imblearn.over_sampling import SMOTE

# Load dataset
data = pd.read_csv('/mnt/data/preprocessedfinal_mental_health_data_standardized.csv')


# Drop non-numeric columns and target
X = data.drop(columns=['index', 'Entity', 'Code', 'Year', 'Depression (%)'])
y = (data['Depression (%)'] > data['Depression (%)'].median()).astype(int)

# Feature Engineering
X['Schizophrenia_Bipolar'] = X['Schizophrenia (%)'] * X['Bipolar disorder (%)']
X['Anxiety_Drug'] = X['Anxiety disorders (%)'] * X['Drug use disorders (%)']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the classes in the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

def run_tuning_page():
    st.title("Tune LightGBM Model Parameters")

    # Tuning sliders for key parameters
    n_estimators = st.slider("Number of Boosting Rounds (n_estimators)", 10, 150, 50)
    learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1)
    max_depth = st.slider("Max Depth", -1, 20, -1)

    # Train model with adjusted parameters
    model = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
    model.fit(X_train_smote, y_train_smote)

    # Evaluate the model
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    # Display metrics
    st.write(f"Test AUC: {auc:.4f}")
    st.write(f"Test Accuracy: {model.score(X_test, y_test) * 100:.2f}%")

