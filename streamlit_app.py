import streamlit as st
from tuning import run_tuning_page

# Main App
st.sidebar.title("|||")
page = st.sidebar.radio("Models", ["Model Tuning"])

# Routing to the pages
if page == "Model Tuning":
    run_tuning_page()
