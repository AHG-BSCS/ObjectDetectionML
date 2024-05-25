import ui

import os
import cv2
import pickle
import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.models import resnet18

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load a pre-trained ResNet model for feature extraction
resnet_model = resnet18(pretrained=True)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])
resnet_model.eval()

# Define the transform to preprocess the image for ResNet
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # ResNet expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image):
    # Apply the preprocessing transform
    input_tensor = preprocess(image)
    # Add a batch dimension
    input_tensor = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        # Extract features
        features = resnet_model(input_tensor)
    
    # Flatten the features to a 1D vector
    features = features.view(features.size(0), -1)
    return features.numpy().flatten()

# Function to load images and labels from the dataset directory
def load_dataset(dataset_path, target_size=(224, 224)):
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

def save_model_and_data(knn, X_train, y_train, label_to_int, int_to_label, filename='trained_model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump((knn, X_train, y_train, label_to_int, int_to_label), file)

def load_model_and_data(filename='trained_model.pkl'):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def main():
    dataset_path = 'dataset'  # Update this to your actual dataset path

    # Check if trained model and data already exist
    if os.path.exists('trained_model.pkl'):
        knn, X_train, y_train, label_to_int, int_to_label = load_model_and_data()
        print("Model and data loaded from file.")
    else:
        # Load dataset
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

        # Save the trained model and data
        save_model_and_data(knn, X_train, y_train, label_to_int, int_to_label)
        print("Model and data saved to file.")

    # Start the GUI
    ui.start_gui(knn, y_train, int_to_label)