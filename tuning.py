import streamlit as st
from model import lgb_model, X_train_smote, y_train_smote, X_test, y_test
from performance import display_performance_metrics
from lime_analysis import display_lime_analysis

# Tuning Page
def run_tuning_page():
    st.title("LightGBM Model Tuning for Understanding Treatment of Depression Disorders")

    # Add subtitles below the title
    st.write("Name: Joel Ling Shern TP no: TP072975")
    st.write("Supervisor: Dr. Minnu Helen Joseph")
    st.write("2nd marker: Assoc. Prof. Dr. Nirase Fathima Abubacker")
    st.write("") 
    st.write("")  
    
    # Tuning sliders for key parameters
    n_estimators = st.slider("Number of Boosting Rounds (n_estimators)", 10, 150, 50)
    learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1)
    max_depth = st.slider("Max Depth", -1, 20, -1)

    # Update the model with tuned parameters
    lgb_model.set_params(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
    lgb_model.fit(X_train_smote, y_train_smote)

    # Sidebar navigation for performance metrics or LIME analysis
    st.subheader("View Results")
    option = st.radio("Select what to view", ["Performance Metrics", "LIME Analysis"])

    # Routing to display metrics or LIME analysis
    if option == "Performance Metrics":
        display_performance_metrics(lgb_model)
    elif option == "LIME Analysis":
        display_lime_analysis(lgb_model)
