import os
import pickle
import numpy as np
from preprocess import load_data
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading dataset...")

X, y = load_data(r"C:\ML_Datasets\dogs_vs_cats")
                 
print("Total samples:", len(X))

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training SVM...")

model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# metrics
acc = accuracy_score(y_test, y_pred)

print("\nAccuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# save everything for dashboard
os.makedirs("model", exist_ok=True)

pickle.dump(model, open("model/svm_model.pkl", "wb"))
pickle.dump((y_test, y_pred), open("model/eval.pkl", "wb"))

print("\nModel + evaluation saved!")