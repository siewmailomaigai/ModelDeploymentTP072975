import streamlit as st
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

# Function to display LIME Analysis
def display_lime_analysis(model):
    st.subheader("LIME Interpretability Analysis")

    # LIME analysis for interpretability
    explainer = LimeTabularExplainer(X_train_smote.values, feature_names=X_train.columns, class_names=['Low', 'High'], mode='classification')

    # Loop through a few instances and display LIME explanations
    for instance_idx in [0, 5, 10, 15, 20]:
        st.write(f"Explanation for instance {instance_idx}:")
        exp = explainer.explain_instance(X_test.iloc[instance_idx].values, model.predict_proba, num_features=len(X_test.columns))
        fig = exp.as_pyplot_figure()
        st.pyplot(fig)
