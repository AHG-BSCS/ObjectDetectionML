import os
import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load a pre-trained ResNet model for feature extraction
resnet_model = resnet18(pretrained=True)
resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])
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

# Function to start the camera and object detection
def start_camera():
    cap = cv2.VideoCapture(0)

    # Set initial window size (adjust as needed)
    initial_window_size = (640, 480)
    cv2.namedWindow('Live Object Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Live Object Detection', *initial_window_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform inference with YOLOv5
        results = model(frame)

        # Process detections
        for *box, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, box)
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:  # Check if ROI is valid
                roi_resized = cv2.resize(roi, (224, 224))  # ResNet expects 224x224 images
                
                # Extract features from the ROI using the ResNet model
                features = extract_features(roi_resized)
                features = features.reshape(1, -1)
                
                # Use k-NN to classify the object based on extracted features
                neighbors = knn.kneighbors(features, return_distance=False)
                
                # Get the most common label among the neighbors
                neighbor_labels = y_train[neighbors[0]]
                unique, counts = np.unique(neighbor_labels, return_counts=True)
                most_common_label = unique[np.argmax(counts)]
                
                # Set a confidence threshold
                confidence_threshold = 0.6
                confidence = np.max(counts) / knn.n_neighbors
                
                if confidence > confidence_threshold:
                    predicted_label = int_to_label[most_common_label]
                    # Draw bounding box around the detected object and display the prediction
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Prediction: {predicted_label}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Check if the window is still open
        if cv2.getWindowProperty('Live Object Detection', cv2.WND_PROP_VISIBLE) < 1:
            break

        # Get window size         
        window_h, window_w = cv2.getWindowImageRect('Live Object Detection')[3], cv2.getWindowImageRect('Live Object Detection')[2]

        # Calculate aspect ratio of the frame
        frame_h, frame_w = frame.shape[:2]
        aspect_ratio = frame_w / frame_h

        # Calculate new dimensions to fit the frame within the window without stretching
        if window_w != 0 or window_h != 0:
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
        
        if cv2.waitKey(1) & 0xFF == ord('v'):
            verify_dataset()

    cap.release()
    cv2.destroyAllWindows()

# Function to start the GUI
def start_gui():
    def on_start_button():
        root.destroy()
        start_camera()

    root = tk.Tk()
    root.title("Object Detection")

    # Set window size
    window_width = 400
    window_height = 300

    # Get screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Calculate position x, y to center the window
    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)

    # Set the position of the window to the center of the screen
    root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

    # Disable maximize button and make the window non-resizable
    root.resizable(False, False)

    # Load the background image using a relative path
    script_dir = os.path.dirname(__file__)  # Absolute path to the directory of the script
    image_path = os.path.join(script_dir, 'assets', 'background_image.jpg')  # Relative path to the image
    original_image = Image.open(image_path)

    # Resize the image to fit the window size
    resized_image = original_image.resize((window_width, window_height), Image.LANCZOS)
    background_photo = ImageTk.PhotoImage(resized_image)

    # Create a canvas and add the background image
    canvas = Canvas(root, width=window_width, height=window_height)
    canvas.pack(fill='both', expand=True)
    image_id = canvas.create_image(0, 0, image=background_photo, anchor='nw')
    canvas.image = background_photo  # Keep a reference to the image to prevent garbage collection

    # Add the label and button on the canvas
    label = tk.Label(root, text="Welcome to Object Detection System", font=("Times New Roman", 18), bg='white')
    label_window = canvas.create_window(window_width // 2, window_height // 3, anchor='center', window=label)

    start_button = tk.Button(root, text="Start", font=("Arial", 16), command=on_start_button)
    button_window = canvas.create_window(window_width // 2, window_height // 2, anchor='center', window=start_button)

    root.mainloop()

# Function to display dataset images used during training with bounding boxes
def verify_dataset():
    # Create a new window
    dataset_window = tk.Toplevel()
    dataset_window.title("Dataset Images with Bounding Boxes")

    # Set window size
    window_width = 800
    window_height = 600
    dataset_window.geometry(f'{window_width}x{window_height}')

    # Load the dataset images used during training
    dataset_images = []
    for label in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, label)
        if os.path.isdir(class_dir):
            class_images = [cv2.imread(os.path.join(class_dir, img_name)) for img_name in os.listdir(class_dir)]
            dataset_images.extend(class_images)

    # Function to display next image upon pressing Enter
    def next_image(event):
        nonlocal idx
        idx += 1
        if idx < len(dataset_images):
            img = dataset_images[idx]
            results = model(img)

            # Process detections
            for *box, conf, cls in results.xyxy[0]:
                x1, y1, x2, y2 = map(int, box)
                # Draw bounding box on the image
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Convert the image to RGB format
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Convert the image to PIL format
            pil_img = Image.fromarray(img_rgb)
            # Resize the image to fit the window
            pil_img_resized = pil_img.resize((window_width // 2, window_height // 2), Image.LANCZOS)
            # Convert PIL image to Tkinter PhotoImage
            photo = ImageTk.PhotoImage(pil_img_resized)

            # Update the image label
            label_img.configure(image=photo)
            label_img.image = photo  # Keep a reference to the image to prevent garbage collection
            label_img.pack()

    # Initialize index to track current image
    idx = 0

    # Display the first image
    img = dataset_images[idx]
    results = model(img)

    # Process detections
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        # Draw bounding box on the image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Convert the image to RGB format
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert the image to PIL format
    pil_img = Image.fromarray(img_rgb)
    # Resize the image to fit the window
    pil_img_resized = pil_img.resize((window_width // 2, window_height // 2), Image.LANCZOS)
    # Convert PIL image to Tkinter PhotoImage
    photo = ImageTk.PhotoImage(pil_img_resized)

    # Create a label to display the image
    label_img = tk.Label(dataset_window, image=photo)
    label_img.image = photo  # Keep a reference to the image to prevent garbage collection
    label_img.pack()

    # Bind the Enter key to display the next image
    dataset_window.bind('<Return>', next_image)

    dataset_window.mainloop()

# Main script
if __name__ == "__main__":
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

    # Start the GUI
    start_gui()
