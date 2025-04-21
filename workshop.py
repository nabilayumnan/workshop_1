## NABILA YUMNA NAAFI'A
## 23/511456/PA/21799
#================================================================================================================================

import cv2
import numpy as np
import pickle
from sklearn.base import BaseEstimator, TransformerMixin

# Define the MeanCentering class
class MeanCentering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mean_face = np.mean(X, axis=0)
        return self

    def transform(self, X):
        return X - self.mean_face

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the trained Eigenface model
with open('eigen_face_assignment.pkl', 'rb') as f:
    pipe = pickle.load(f)

# Function to detect faces in a frame
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces, gray

# Function to predict the label for a cropped face
def predict_label(face):
    face_resized = cv2.resize(face, (128, 128))  # Resize to match the training size
    face_flattened = face_resized.flatten().reshape(1, -1)  # Flatten and reshape for prediction
    label = pipe.predict(face_flattened)[0]  # Predict the label
    return label

# Initialize the camera
camera = cv2.VideoCapture(0)

print("Starting face detection with predictions. Press 'q' to quit.")
while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Detect faces in the frame
    faces, gray = detect_faces(frame)

    # Draw rectangles around detected faces and add predicted labels
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]  # Crop the face
        label = predict_label(face)  # Predict the label using the Eigenface model
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Green box with thickness 3
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  # Add label above the box

    # Display the frame with detected faces and labels
    cv2.imshow("Face Detection with Predictions", frame)

    # Exit the loop when 'q' is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()