# 🍽️ Food 101 Recreation Project 🍴  

This project focuses on classifying food images into different categories using machine learning. Using the **Food 101 dataset** and incorporating 10 different food classes, this project demonstrates the power of computer vision in recognizing and categorizing food items.  

---

## 🚀 Project Overview  
- **Objective**: Classify food items into 10 distinct categories based on images using machine learning techniques.
- **Dataset**: Food 101 dataset containing images of 101 food categories, but for this project, we are focusing on **10 specific food classes**.
- **Approach**:  
  - Preprocess food images (resizing, normalization).  
  - Train convolutional neural networks (CNNs) on the images.  
  - Evaluate model performance using accuracy, loss metrics, and confusion matrices.  

---

## 🛠️ Tools and Technologies  
- **Python** 🐍  
- **TensorFlow/Keras** for deep learning model development  
- **OpenCV** for image processing  
- **NumPy** and **Pandas** for data handling  
- **Matplotlib** and **Seaborn** for data visualization  
- **Jupyter Notebook** for development  

---

## 📊 Dataset Details  
- **Total classes**: 10 food categories  
- **Food classes used in this project**:  
  - Pizza  
  - Burger  
  - Sandwich  
  - Sushi  
  - Cake  
  - Donut  
  - Ice Cream  
  - Salad  
  - Fruit  
  - Pasta  

- **Image size**: Each image is resized to a consistent size (e.g., 128x128 pixels) for input into the model.  
- **Source**: The dataset is derived from the Food 101 dataset available on Kaggle.  

---

## 🧠 Model Overview  
1. **Model Architecture**:  
   - A Convolutional Neural Network (CNN) with multiple layers for image classification.
   - Includes layers such as convolution, pooling, and dense layers to extract features from the images.
   - Utilizes **softmax** activation function in the output layer for multi-class classification.

2. **Training**:  
   - The model is trained using categorical cross-entropy loss function and Adam optimizer for efficient training.
   - The dataset is split into training, validation, and test sets for robust evaluation.

---

## 🏆 Results  
- **Accuracy**: The model achieved an accuracy of **XX%** on the validation dataset.
- **Confusion Matrix**: A confusion matrix was generated to evaluate how well the model classified each food item.
- **Key performance metrics**: Loss, accuracy, and F1-Score were used for model evaluation.

---
