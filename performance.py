import streamlit as st
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from model import X_test, y_test

def display_performance_metrics(model):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Output metrics
    st.write(f"Test Accuracy: {accuracy * 100:.2f}%")
    st.write(f"AUC: {auc:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=True,
                xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for LightGBM Classifier')
    plt.show()
