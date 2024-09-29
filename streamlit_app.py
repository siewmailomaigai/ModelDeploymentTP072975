import streamlit as st
from tuning import run_tuning_page

# Load image icons
image_1 = "Sustainable_Development_Goal_03GoodHealth.svg"
image_2 = "APUlogo.jpg"

# Add image icons at the top left with minimal spacing
col1, col2 = st.columns([0.5, 0.5])  # Reduce the column width to bring them closer
with col1:
    st.image(image_1, width=150)

with col2:
    st.image(image_2, width=150)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Models", ["Model Tuning"])

# Routing to the pages
if page == "Model Tuning":
    run_tuning_page()
