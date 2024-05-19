import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Function to extract features (simple grayscale flattening for this example)
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (50, 50))  # Resize to 50x50 pixels
    flattened = resized.flatten()
    return flattened

# Function to load images and labels from the dataset directory
def load_dataset(dataset_path, target_size=(100, 100)):
    images = []
    labels = []
    for label in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, label)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    # Resize image to target size
                    img = cv2.resize(img, target_size)
                    images.append(img)
                    labels.append(label)  # Use label names as the labels
    return images, labels


# Load dataset
dataset_path = 'dataset'  # Update this to your actual dataset path
images, labels = load_dataset(dataset_path)

# Convert string labels to integers
unique_labels = list(set(labels))
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
int_to_label = {idx: label for label, idx in label_to_int.items()}  # Reverse mapping
y = np.array([label_to_int[label] for label in labels])

# Extract features and labels
X = np.array([extract_features(img) for img in images])

# Split the dataset into training and test sets for validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train k-NN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluate on test set
accuracy = knn.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Set up live camera feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize frame to the same size as training images (for simplicity)
    frame_resized = cv2.resize(frame, (50, 50))
    
    # Extract features from the frame
    features = extract_features(frame_resized)
    
    # Reshape and predict
    features = features.reshape(1, -1)
    prediction = knn.predict(features)
    
    # Convert integer prediction back to label
    predicted_label = int_to_label[prediction[0]]
    
    # Draw bounding box around the object and display the prediction
    height, width, _ = frame.shape
    cv2.rectangle(frame, (10, 10), (width - 10, height - 10), (0, 255, 0), 2)
    cv2.putText(frame, f"Prediction: {predicted_label}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow('Live Object Detection', frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
