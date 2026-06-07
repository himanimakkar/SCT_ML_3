# 🐾 SCT_ML_3: Cat vs Dog Image Classification using SVM

## 📌 Overview
This project is a machine learning-based image classification system that distinguishes between cats and dogs using a Support Vector Machine (SVM) model. It uses Histogram of Oriented Gradients (HOG) for feature extraction and provides a Streamlit-based web application for real-time predictions.

---

## 🎯 Objective
To build an end-to-end ML pipeline that classifies images of cats and dogs using classical machine learning techniques.

---

## 📂 Project Structure
SCT_ML_3/
│
├── app/ # Streamlit web application
├── dataset/ # Dataset (excluded using .gitignore)
├── model/ # Trained SVM model
├── preprocess.py # Feature extraction & preprocessing
├── train.py # Model training script
├── requirements.txt # Dependencies
├── .gitignore
└── README.md

---

## ⚙️ Tech Stack
- Python 🐍
- OpenCV 👁️
- Scikit-learn 🤖
- NumPy & Pandas 📊
- HOG Feature Extraction
- Streamlit (UI)

---

## 🧠 Machine Learning Pipeline

1. Load dataset (Cats & Dogs images)
2. Resize images to standard format
3. Convert images to grayscale
4. Extract HOG features
5. Train SVM classifier
6. Evaluate model performance
7. Save trained model
8. Deploy using Streamlit UI

---

## 📊 Model Performance

- Algorithm: Support Vector Machine (SVM)
- Feature Extraction: HOG
- Accuracy: ~60–75% (depends on dataset size)

---

## 🖥️ Web App Features

✔ Upload image (Cat/Dog)  
✔ Real-time prediction  
✔ Confidence score  
✔ Clean dashboard UI  
✔ Interactive Streamlit interface  

---

## 🚀 How to Run the Project

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
### 2️⃣ Train the model
```bash
python train.py
### 3️⃣ Run the web app
```bash
streamlit run app/app.py
