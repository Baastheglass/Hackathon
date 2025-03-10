# 🩺 Diabetic Retinopathy Detection  

Model: https://drive.google.com/file/d/1TDeVDD5sUQ1sJlSt2tDmTmE12fxANQOG/view?usp=sharing

This project is an AI-powered application designed to classify **retinal fundus images**
into different **Diabetic Retinopathy (DR) severity levels**. The model processes the uploaded
images and predicts the severity of DR using deep learning. The frontend is built using
**Streamlit** for an interactive and user-friendly experience.

---
👨‍💻 Authors
Team Name - Baasil&Hafsa
Developer1 - Muhammad Baasil
Developer2 - Hafsa Wajid Butt

---
## 🚀 Features  
✅ Upload retinal images for analysis  
✅ AI model predicts the DR severity level  
✅ Results displayed as a **percentage bar** and **pie chart**  

## 📈 Usage Instructions
1. Clone the repository to your local machine.
2. Install the required packages using pip.
3. Download the model from the provided link and put it into the model folder.
4. Run the application using `streamlit run ./code/frontend.py` in your terminal.
5. Upload a retinal fundus image to the application.
6. The AI model will process the image and display the predicted DR severity level as a 
percentage bar and a pie chart.
---

## 📊 Model Architecture
The model is a **Convolutional Neural Network (CNN)** with the following architecture:

4 convolutional layers (32, 64, 128, 128 filters) with ReLU activation.
4 max pooling layers (2x2).
A fully connected dense layer with 512 neurons and ReLU activation.
A dropout layer (0.5) to prevent overfitting.
A final output layer with 5 neurons and softmax activation for classification.

---
## 📊 Model Training
The model is trained on the **DRISHTI-GS** dataset, which contains 5000
images of retinal fundus. The dataset is split into training and testing sets with a ratio of
70:30. The model is trained using the **ADABoost** with a learning rate of
0.001 and a batch size of 32. The model is trained for 5 epochs with
early stopping. The model achieves a **test accuracy of 0.49** We could get more accuracy but 
due to time constraint we only get this acuracy 
