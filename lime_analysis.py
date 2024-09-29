from model import X_train_smote, X_test

def display_lime_analysis(model):
    explainer = LimeTabularExplainer(X_train_smote.values, feature_names=X_train_smote.columns, class_names=['Low', 'High'], mode='classification')

    for instance_idx in [0, 5, 10, 15, 20]:  # Adjust based on instances of interest
        exp = explainer.explain_instance(X_test.iloc[instance_idx].values, model.predict_proba, num_features=len(X_test.columns))

        # Plot the LIME explanation for feature importance
        plt.figure(figsize=(6, 8))
        exp.as_pyplot_figure()
        plt.tight_layout()
        plt.show()
