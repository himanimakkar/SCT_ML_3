# SCT_ML_3: Cat vs Dog Image Classification using SVM

## 📌 Objective
To classify images of cats and dogs using Support Vector Machine (SVM) with HOG feature extraction.

## 📊 Dataset
Kaggle Dogs vs Cats dataset

## ⚙️ Tech Stack
- Python
- OpenCV
- Scikit-learn
- HOG (Feature Extraction)
- Streamlit (UI)

## 🧠 Model Workflow
Image → Resize → Grayscale → HOG Features → SVM → Prediction

## 🚀 How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Train model
python train.py

### 3. Run UI
streamlit run app/app.py

## 📈 Output
Predicts whether uploaded image is a Cat 🐱 or Dog 🐶