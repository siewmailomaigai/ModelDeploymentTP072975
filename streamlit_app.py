import streamlit as st
from tune import run_tuning_page
from lime_analysis import run_lime_analysis

# Sidebar navigation
st.sidebar.title("LightGBM Mental Health Prediction App")
page = st.sidebar.selectbox("Navigation", ["Model Tuning", "LIME Interpretability Analysis"])

# Routing to different pages
if page == "Model Tuning":
    run_tuning_page()
elif page == "LIME Interpretability Analysis":
    run_lime_analysis()
