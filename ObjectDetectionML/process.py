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

# Loads pretrained yolov5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Loads pretrained resnet model
resnet_model = resnet18(pretrained=True)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])
resnet_model.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Preprocesses and extract features from images
def extract_features(image):
    input_tensor = preprocess(image)
    input_tensor = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        features = resnet_model(input_tensor)
    
    features = features.view(features.size(0), -1)
    return features.numpy().flatten()

# Loads the dataset
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
            print(f"Loaded images for {label}.")
    return images, labels

# Saves data as a file
def save_model_and_data(knn, X_train, y_train, label_to_int, int_to_label, filename='trained_model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump((knn, X_train, y_train, label_to_int, int_to_label), file)

# Loads data from existing file
def load_model_and_data(filename='trained_model.pkl'):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Updates data using the latest dataset
def update_dataset(dataset_path='dataset'):
    print("Loading data...")
    images, labels = load_dataset(dataset_path)

    print("")
    print("Updating data please wait...")
    print("")
    print("This may take some time... (DON'T CLOSE THIS WINDOW)")
    print("")
    
    unique_labels = list(set(labels))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    int_to_label = {idx: label for label, idx in label_to_int.items()}
    y = np.array([label_to_int[label] for label in labels])

    X = np.array([extract_features(img) for img in images])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    accuracy = knn.score(X_test, y_test)

    save_model_and_data(knn, X_train, y_train, label_to_int, int_to_label)




def main():
    if os.path.exists('trained_model.pkl'):
        knn, X_train, y_train, label_to_int, int_to_label = load_model_and_data()
    else:
        update_dataset()
        
    # Start the GUI
    ui.start_gui(knn, y_train, int_to_label)