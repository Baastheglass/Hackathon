import streamlit as st
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
from use_dr_model import predict_image, load_pretrained_model
# Set page configuration
st.set_page_config(page_title="Diabetic Retinopathy Detection", page_icon="🩺", layout="centered")

#CSS for colors
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

# Title
st.markdown("<h1 style='color: #C599B6; text-align: center;'>Diabetic Retinopathy Detection</h1>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload a retinal image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    temp_file_path = "temp_image.jpg"
    # Save the uploaded file to a temporary location
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Now you can use the file path
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Processing indicator
    with st.spinner("Analyzing image..."):
        model = load_pretrained_model("./model/diabetic_retinopathy_cnn_model.pth")
        print(image)
        predicted_class, predicted_percentages = predict_image(model, temp_file_path)
        print("Predicted Class: ", predicted_class)
        print("Predicted Percentages: ", predicted_percentages)
        average = np.mean(predicted_percentages)
        severity_percentage = average 
    
    # Display result
    st.markdown(f"<h2 style='color: #D8A0A8;'>Severity Level: {severity_percentage}%</h2>", unsafe_allow_html=True)
    st.progress(severity_percentage / 100)  #progress bar
    
    #severity as a pie chart
    fig, ax = plt.subplots()
    ax.pie([severity_percentage, 100 - severity_percentage], labels=["DR Severity", "Healthy"], colors=["#EAB8B8", "#FFD1C1"], autopct='%1.1f%%')
    st.pyplot(fig)
    
    #reprocess button
    if st.button("Analyze Another Image"):
        st.experimental_rerun()
