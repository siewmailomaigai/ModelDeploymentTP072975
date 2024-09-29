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

    # Loop through selected instances
    for instance_idx in [0, 5, 10, 15, 20]:  # Adjust based on instances of interest
        st.write(f"Explanation for instance {instance_idx}:")
        
        # Explain the instance prediction
        exp = explainer.explain_instance(
            X_test.iloc[instance_idx].values, 
            model.predict_proba, 
            num_features=len(X_train_smote.columns)  # Show all features
        )

        # Plot the LIME explanation for feature importance
        plt.figure(figsize=(6, 8))
        exp.as_pyplot_figure()  # Generate the LIME plot
        plt.tight_layout()  # Ensure the plot doesn't overlap
        st.pyplot(plt)  # Use Streamlit's display function for plots

        # Extract the top 3 most important features
        top_3_features = sorted(exp.as_list(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        # Print the top 3 feature importances below the plot
        st.write("**Top 3 Important Features for this instance**:")
        for feature, importance in top_3_features:
            st.write(f"- {feature}: {importance:.4f}")
