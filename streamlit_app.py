import streamlit as st
from tuning import run_tuning_page

# Load image icons
image_1 = "Sustainable_Development_Goal_03GoodHealth.svg"  # Replace with actual file path or URL
image_2 = "path_to_your_image_2.png"  # Replace with actual file path or URL

# Add image icons at the top left
col1, col2 = st.columns([1, 9])  # Create two columns
with col1:
    st.image(image_1, width=50)  # Adjust width for desired size
    st.image(image_2, width=50)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Models", ["Model Tuning"])

# Routing to the pages
if page == "Model Tuning":
    run_tuning_page()
