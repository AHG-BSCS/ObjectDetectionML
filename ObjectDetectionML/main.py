import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Function to extract HOG features
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (50, 50))  # Ensure consistent size
    features, hog_image = hog(resized, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), 
                              visualize=True, feature_vector=True)
    return features

# Function to load images and labels from the dataset directory
def load_dataset(dataset_path, target_size=(50, 50)):
    images = []
    labels = []
    for label in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, label)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, target_size)
                    images.append(img)
                    labels.append(label)
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
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Evaluate on test set
accuracy = knn.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Set up live camera feed
cap = cv2.VideoCapture(0)

# Set initial window size (adjust as needed)
initial_window_size = (640, 480)
cv2.namedWindow('Live Object Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Live Object Detection', *initial_window_size)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box for the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Extract the ROI and resize it to the target size
        roi = frame[y:y+h, x:x+w]
        if roi.size > 0:  # Check if ROI is valid
            roi_resized = cv2.resize(roi, (50, 50))
            
            # Extract features from the ROI
            features = extract_features(roi_resized)
            features = features.reshape(1, -1)
            neighbors = knn.kneighbors(features, return_distance=False)
            
            # Get the most common label among the neighbors
            neighbor_labels = y_train[neighbors[0]]
            (unique, counts) = np.unique(neighbor_labels, return_counts=True)
            most_common_label = unique[np.argmax(counts)]
            
            # Set a confidence threshold
            confidence_threshold = 0.6
            confidence = np.max(counts) / knn.n_neighbors
            
            if confidence > confidence_threshold:
                predicted_label = int_to_label[most_common_label]
                # Draw bounding box around the detected object and display the prediction
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Prediction: {predicted_label}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Get window size
    window_h, window_w = cv2.getWindowImageRect('Live Object Detection')[3], cv2.getWindowImageRect('Live Object Detection')[2]
    
    # Calculate aspect ratio of the frame
    frame_h, frame_w = frame.shape[:2]
    aspect_ratio = frame_w / frame_h
    
    # Calculate new dimensions to fit the frame within the window without stretching
    if window_w / window_h > aspect_ratio:
        new_w = int(window_h * aspect_ratio)
        new_h = window_h
    else:
        new_h = int(window_w / aspect_ratio)
        new_w = window_w
    
    # Resize frame to fit within the window
    resized_frame = cv2.resize(frame, (new_w, new_h))
    
    # Show the frame with black borders to maintain aspect ratio
    border_w = (window_w - new_w) // 2
    border_h = (window_h - new_h) // 2
    bordered_frame = cv2.copyMakeBorder(resized_frame, border_h, border_h, border_w, border_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    # Show the frame
    cv2.imshow('Live Object Detection', bordered_frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
