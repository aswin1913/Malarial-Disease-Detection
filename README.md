# Malaria Disease Prediction Using Deep Learning

## 📌 Project Overview
This project focuses on detecting **Malaria-infected red blood cells (RBCs)** using a **Convolutional Neural Network (CNN)**. The model classifies images as either **Parasitized (Infected)** or **Unparasitized (Healthy)** based on microscopic blood smear images.

## 🚀 Features
- Uses **Deep Learning (CNN)** for malaria detection.
- **Automated image classification** of malaria-infected cells.
- **Trained using TensorFlow and Keras**.
- **Dataset Preprocessing and Augmentation** to improve accuracy.
- **User-Friendly Image Upload and Prediction**.

## 📂 Dataset
The dataset consists of **two categories**:
1. **Parasitized** - RBCs affected by Malaria.
2. **Unparasitized** - Healthy RBCs.

The dataset is to be available in your google drive.

## 🛠️ Technologies Used
- **Python** 🐍
- **TensorFlow & Keras** 🔥
- **OpenCV & Matplotlib** 📊
- **Google Colab (Training)** 💻

## 🔧 Installation
###  Clone the Repository  

git clone https://github.com/aswin1913/Malarial-Disease-Detection.git
Malarial-Disease-Detection

## Install Dependencies
pip install tensorflow numpy matplotlib opencv-python

## Run Model Training
python train_model.py

## Run Image Prediction
Upload an image and execute:
python predict.py --image sample.png

## 📊 Model Architecture
Conv2D + MaxPooling layers for feature extraction.
Flatten + Dense layers for classification.
Softmax activation for final prediction.

## 📈 Training Performance
Achieved high accuracy (>95%) on the test dataset.
Faster and more consistent diagnosis compared to manual methods.

## 🤖 Future Improvements
Enhance model accuracy with data augmentation.
Deploy as a web application for real-time detection.
Extend model for other blood cell diseases.
