# 🩺 Diabetic Retinopathy Detection  

This project is an AI-powered application designed to classify **retinal fundus images**
into different **Diabetic Retinopathy (DR) severity levels**. The model processes the uploaded
images and predicts the severity of DR using deep learning. The frontend is built using
**Streamlit** for an interactive and user-friendly experience.

---

## 🚀 Features  
✅ Upload retinal images for analysis  
✅ AI model predicts the DR severity level  
✅ Results displayed as a **percentage bar** and **pie chart**  

things to Install before RUN

pip install Streamlit
pip install tensorflow
pip install numpy
pip install matplotlib
pip install pandas
pip install torch
pip install torchvision
pip install scikit-learn
pip install opencv-python
pip install Pillow
---

## 📈 Usage Instructions
1. Clone the repository to your local machine.
2. Install the required packages using pip.
3. Run the application using `streamlit run ./code/frontend.py` in your terminal.
4. Upload a retinal fundus image to the application.
5. The AI model will process the image and display the predicted DR severity level as a 
percentage bar and a pie chart.
---


## 📊 Model Architecture
The model is a **Convolutional Neural Network (CNN)** with the following architecture:
- **Conv2D** layer with 32 filters, kernel size 3, and ReLU
- **MaxPooling2D** layer with pool size 2
- **Flatten** layer
- **Dense** layer with 128 units and ReLU
- **Dense** layer with 1 unit and sigmoid activation
---
## 📊 Model Training
The model is trained on the **DRISHTI-GS** dataset, which contains 88
images of retinal fundus. The dataset is split into training and testing sets with a ratio of
80:20. The model is trained using the **Adam optimizer** with a learning rate of
0.001 and a batch size of 32. The model is trained for 5 epochs with
early stopping. The model achieves a **test accuracy of 0.95** and a **test
loss of 0.05**.
