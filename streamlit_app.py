import streamlit as st
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from lime.lime_tabular import LimeTabularExplainer

# Load dataset
data = pd.read_csv('preprocessedfinal_mental_health_data_standardized.csv')

# Drop non-numeric columns and target
X = data.drop(columns=['index', 'Entity', 'Code', 'Year', 'Depression (%)'])
y = (data['Depression (%)'] > data['Depression (%)'].median()).astype(int)

# Feature Engineering
X['Schizophrenia_Bipolar'] = X['Schizophrenia (%)'] * X['Bipolar disorder (%)']
X['Anxiety_Drug'] = X['Anxiety disorders (%)'] * X['Drug use disorders (%)']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance the classes in the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Function to run the main page with model training and results
def run_model_page():
    st.title("LightGBM Model Training & Evaluation")

    # Tuning sliders for key parameters
    n_estimators = st.slider("Number of Boosting Rounds (n_estimators)", 10, 150, 50)
    learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1)
    max_depth = st.slider("Max Depth", -1, 20, -1)

    # Build the LightGBM model
    lgb_model = lgb.LGBMClassifier(
        boosting_type='gbdt',
        num_leaves=31,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=42
    )

    # Train the model
    lgb_model.fit(X_train_smote, y_train_smote)

    # Save the model
    with open('lightgbm_model.pkl', 'wb') as file:
        pickle.dump(lgb_model, file)

    # Perform Cross-Validation
    score = cross_val_score(lgb_model, X_train_smote, y_train_smote, scoring='accuracy', cv=5, n_jobs=-1)
    st.write(f"Cross-Validation Accuracy: {score.mean() * 100:.2f}% with a standard deviation of {score.std() * 100:.2f}")

    # Training Evaluation
    train_ypred = lgb_model.predict(X_train_smote)
    train_accuracy = accuracy_score(y_train_smote, train_ypred)
    st.write(f"Training Accuracy: {train_accuracy * 100:.2f}%")

    # Testing Evaluation
    test_ypred = lgb_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_ypred)
    st.write(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

    # AUC, Recall, F1 Score for the test set
    y_prob = lgb_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, test_ypred)
    f1 = f1_score(y_test, test_ypred)

    st.write(f"AUC: {auc:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")

    # Classification Report
    st.write("Classification Report:")
    st.text(classification_report(y_test, test_ypred, target_names=['Low', 'High']))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, test_ypred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=True,
                xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for LightGBM Classifier')
    st.pyplot(plt)

    # LIME Analysis for Interpretability
    st.write("LIME Interpretability Analysis for 5 Instances")
    explainer = LimeTabularExplainer(X_train_smote.values, feature_names=X_train.columns, class_names=['Low', 'High'], mode='classification')

    for instance_idx in [0, 5, 10, 15, 20]:
        st.write(f"Explanation for instance {instance_idx}:")
        exp = explainer.explain_instance(X_test.iloc[instance_idx].values, lgb_model.predict_proba, num_features=len(X_test.columns))
        fig = exp.as_pyplot_figure()
        st.pyplot(fig)

# Sidebar navigation for switching between pages
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Model Training", "LIME Analysis"])

if page == "Model Training":
    run_model_page()
