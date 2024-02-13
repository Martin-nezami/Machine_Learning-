import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Function to extract features from images
def extract_features(image):
    # Here, we can use simple features like color histogram or HOG features
    # For simplicity, let's use color histograms
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Load images and extract features
def load_images(dataset_path):
    images = []
    labels = []
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                img = cv2.imread(os.path.join(folder_path, filename))
                if img is not None:
                    features = extract_features(img)
                    images.append(features)
                    labels.append(folder_name)  # Use folder name as label
    return images, labels

# Prompt the user to input the path to the dataset
dataset_path = input("Enter the path to the dataset folder: ")

# Load dataset
X, y = load_images(dataset_path)

# Convert labels to integers
label_to_int = {label: idx for idx, label in enumerate(np.unique(y))}
y = [label_to_int[label] for label in y]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create SVM classifier
clf = svm.SVC(kernel='linear')

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Convert predictions back to original labels
int_to_label = {idx: label for label, idx in label_to_int.items()}
y_pred_labels = [int_to_label[idx] for idx in y_pred]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Predict a new image
new_image_path = input("Enter the path to the new image: ")
new_img = cv2.imread(new_image_path)
if new_img is not None:
    new_features = extract_features(new_img)
    predicted_label = clf.predict([new_features])[0]
    predicted_fruit = int_to_label[predicted_label]
    print("Predicted fruit:", predicted_fruit)
else:
    print("Invalid image path or image format.")

