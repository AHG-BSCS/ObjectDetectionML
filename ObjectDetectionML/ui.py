import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk
import process

# Function to start the camera and object detection
def start_camera(knn, y_train, int_to_label):
    cap = cv2.VideoCapture(0)

    # Set initial window size (adjust as needed)
    initial_window_size = (700, 550)
    cv2.namedWindow('Live Object Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Live Object Detection', *initial_window_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform inference with YOLOv5
        results = process.model(frame)

        # Process detections
        for *box, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, box)
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:  # Check if ROI is valid
                roi_resized = cv2.resize(roi, (224, 224))  # ResNet expects 224x224 images
                
                # Extract features from the ROI using the ResNet model
                features = process.extract_features(roi_resized)
                features = features.reshape(1, -1)
                
                # Use k-NN to classify the object based on extracted features
                neighbors = knn.kneighbors(features, return_distance=False)
                
                # Get the most common label among the neighbors
                neighbor_labels = y_train[neighbors[0]]
                unique, counts = np.unique(neighbor_labels, return_counts=True)
                most_common_label = unique[np.argmax(counts)]
                
                confidence = knn.predict_proba(features)[0].max() * 100
                
                if confidence > 60:
                    predicted_label = int_to_label[most_common_label]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                    # Calculate the optimal font scale based on bounding box size
                    box_width = x2 - x1
                    box_height = y2 - y1
                    label_text = f"{predicted_label}: {confidence:.2f}%"
                        
                    font_scale = 1.0
                    font_thickness = 2
                    text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                        
                    while text_size[0] > box_width or text_size[1] > box_height:
                        font_scale -= 0.1
                        text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                        
                    cv2.putText(frame, label_text, (x1, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)

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

    cap.release()
    cv2.destroyAllWindows()

# Function to start the GUI
def start_gui(knn, y_train, int_to_label):
    def on_start_button():
        root.destroy()
        start_camera(knn, y_train, int_to_label)

    def on_update_button():
        root.destroy()
        process.update_dataset()
        process.main()

    def on_quit_button():
        root.destroy()

    root = tk.Tk()
    root.title("Quick Spot")

    # Set window size
    window_width = 700
    window_height = 550

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
    image_path = os.path.join(script_dir, 'assets', 'home_try.png')  # Relative path to the image
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
    #label = tk.Label(root, text="Welcome to Object Detection System", font=("Times New Roman", 18), bg='white')
    #label_window = canvas.create_window(window_width // 2, window_height // 3, anchor='center', window=label)

    button_bg_color = "#AD88C6"
    button_text_color = "#FFE6E6"

    # Place the start button
    start_button = tk.Button(root, text="START", font=("Avenir", 16), command=on_start_button, bg=button_bg_color, fg=button_text_color, cursor="hand2", borderwidth=0)
    start_button.place(x=68, y=183, width=120, height=65)

    # Place the update button
    update_button = tk.Button(root, text="UPDATE", font=("Avenir", 16), command=on_update_button, bg=button_bg_color, fg=button_text_color, cursor="hand2", borderwidth=0)
    update_button.place(x=68, y=258, width=120, height=65)

    # Place the quit button
    quit_button = tk.Button(root, text="QUIT", font=("Avenir", 16), command=on_quit_button, bg=button_bg_color, fg=button_text_color, cursor="hand2", borderwidth=0)
    quit_button.place(x=68, y=333, width=120, height=65)

    root.mainloop()
