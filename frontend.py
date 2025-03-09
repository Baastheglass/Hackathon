import streamlit as st
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Diabetic Retinopathy Detection", page_icon="🩺", layout="centered")

# Apply custom CSS for colors
st.markdown(
    """
    <style>
    body {
        background-color: #FFF7F4; /* Lightest shade */
    }
    .stProgress > div > div > div > div {
        background-color: #EAB8B8 !important; /* Mid-tone shade */
    }
    div.stButton > button:first-child {
        background-color: #C599B6; /* Main theme color */
        color: white;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title with color
st.markdown("<h1 style='color: #C599B6; text-align: center;'>Diabetic Retinopathy Detection</h1>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload a retinal image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Processing indicator
    with st.spinner("Analyzing image..."):
        time.sleep(3)  # Simulate model processing time
        severity_percentage = np.random.randint(0, 101)  # Simulating a model output
    
    # Display result
    st.markdown(f"<h2 style='color: #D8A0A8;'>Severity Level: {severity_percentage}%</h2>", unsafe_allow_html=True)
    st.progress(severity_percentage / 100)  # Display as progress bar
    
    # Show severity as a pie chart
    fig, ax = plt.subplots()
    ax.pie([severity_percentage, 100 - severity_percentage], labels=["DR Severity", "Healthy"], colors=["#EAB8B8", "#FFD1C1"], autopct='%1.1f%%')
    st.pyplot(fig)
    
    # Add a reprocess button
    if st.button("Analyze Another Image"):
        st.experimental_rerun()
