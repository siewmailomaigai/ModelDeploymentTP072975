import streamlit as st
from predict import run_prediction_page
from tune import run_tuning_page

# Sidebar navigation
st.sidebar.title("LightGBM Mental Health Prediction")
page = st.sidebar.selectbox("Navigation", ["Prediction & LIME Analysis", "Model Tuning"])

# Routing to different pages
if page == "Prediction & LIME Analysis":
    run_prediction_page()
elif page == "Model Tuning":
    run_tuning_page()

