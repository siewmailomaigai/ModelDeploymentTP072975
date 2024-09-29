import streamlit as st
from tuning import run_tuning_page

# Load image icons
image_1 = "Sustainable_Development_Goal_03GoodHealth.svg"
image_2 = "APUlogo.jpg"

# Add image icons side by side
st.markdown(
    """
    <style>
    .container {
        display: flex;
        justify-content: flex-start;
        align-items: center;
    }
    .container img {
        margin-right: 20px; /* Adjust margin as necessary */
    }
    </style>
    <div class="container">
        <img src="Sustainable_Development_Goal_03GoodHealth.svg" alt="Image 1" width="150">
        <img src="APUlogo.jpg" alt="Image 2" width="150">
    </div>
    """, 
    unsafe_allow_html=True
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Models", ["Model Tuning"])

# Routing to the pages
if page == "Model Tuning":
    run_tuning_page()
