# Folders to process
sets = ['train', 'val', 'test']
classes = {'NORMAL': 0, 'PNEUMONIA': 1}

# Mouse callback state
ix, iy = -1, -1
import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Parameters
max_images_per_class = 10  # Change as needed
img_size = (64, 64)  # Resize for feature extraction

train_dir = 'train'
classes = {'NORMAL': 0, 'PNEUMONIA': 1}

features = []
labels = []

for label_name, class_id in classes.items():
    image_dir = os.path.join(train_dir, label_name)
    count = 0
    for fname in os.listdir(image_dir):
        if count >= max_images_per_class:
            break
        if fname.lower().endswith(('.jpeg', '.jpg', '.png')):
            img_path = os.path.join(image_dir, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to load {img_path}")
                continue
            img = cv2.resize(img, img_size)
            # Simple features: flatten pixel values
            feat = img.flatten()
            features.append(feat)
            labels.append(class_id)
            count += 1

features = np.array(features)
labels = np.array(labels)

# Train/test split (simple)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)

# Train SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Predict and report
y_pred = clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=list(classes.keys())))