import streamlit as st
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Function to display performance metrics
def display_performance_metrics(model):
    st.subheader("Model Performance Metrics")

    # Testing Evaluation
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

    # AUC, Recall, F1 Score for the test set
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.write(f"AUC: {auc:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")

    # Classification Report
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred, target_names=['Low', 'High']))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=True,
                xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
    st.pyplot(plt)
