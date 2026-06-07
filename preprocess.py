import os
import cv2
import numpy as np
from skimage.feature import hog

# ---------- FEATURE EXTRACTION ----------
def extract_features(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return None

    # resize for speed
    img = cv2.resize(img, (64, 64))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True
    )

    return features


# ---------- LOAD DATA ----------
def load_data(data_dir):
    X = []
    y = []

    cat_count = 0
    dog_count = 0
    max_per_class = 200   # IMPORTANT: keeps it fast

    print("Loading dataset from:", data_dir)

    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)

        # skip folders or invalid files
        if not os.path.isfile(file_path):
            continue

        file_lower = file.lower()

        # CAT CLASS
        if "cat" in file_lower and cat_count < max_per_class:
            feat = extract_features(file_path)
            if feat is not None:
                X.append(feat)
                y.append(0)
                cat_count += 1

        # DOG CLASS
        elif "dog" in file_lower and dog_count < max_per_class:
            feat = extract_features(file_path)
            if feat is not None:
                X.append(feat)
                y.append(1)
                dog_count += 1

        # stop early if both classes are full
        if cat_count >= max_per_class and dog_count >= max_per_class:
            break

    print("Cats loaded:", cat_count)
    print("Dogs loaded:", dog_count)
    print("Total samples:", len(X))

    return np.array(X), np.array(y)