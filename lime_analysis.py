import streamlit as st
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from model import X_train_smote, X_test

def display_lime_analysis(model):
    # Create the LimeTabularExplainer
    explainer = LimeTabularExplainer(
        X_train_smote.values, 
        feature_names=X_train_smote.columns, 
        class_names=['Low', 'High'], 
        mode='classification'
    )

    # Loop through a few example instances
    for instance_idx in [0, 5, 10, 15, 20]:  # Adjust based on instances of interest
        st.write(f"Explanation for instance {instance_idx}:")
        
        # Explain the instance prediction
        exp = explainer.explain_instance(
            X_test.iloc[instance_idx].values, 
            model.predict_proba, 
            num_features=3  # Show the top 3 important features
        )

        # Plot the LIME explanation for feature importance
        plt.figure(figsize=(6, 8))
        exp.as_pyplot_figure()  # Generate the LIME plot
        plt.tight_layout()  # Ensure the plot doesn't overlap
        st.pyplot(plt)  # Use Streamlit's display function for plots

        # Optionally, show the explanation details
        st.write(exp.as_list())  # This shows the top 3 features in a textual format
