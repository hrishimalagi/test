import cv2
import numpy as np
import os
from tkinter import Tk, Label, Button, messagebox, StringVar
from PIL import Image, ImageTk
from tkinter import LabelFrame

# Set the paths for Haar cascade, dataset directory, and trained model file
cascadePath = "haarcascade_frontalface_default.xml"
datasetPath = "Dataset"
trainedModelPath = "trained_model.yml"

# Load the Haar cascade classifier
faceCascade = cv2.CascadeClassifier(cascadePath)

# Create LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Global variables for storing faces and labels
faces = []
labels = []

# Function to train the face recognition model
def train_model():
    global faces, labels

    # Create a dictionary to map unique labels to unique integer IDs
    label_to_id = {}
    current_id = 0

    # Load the dataset and labels
    faces = []
    labels = []
    for root, dirs, files in os.walk(datasetPath):
        for file in files:
            if file.endswith(".jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(os.path.dirname(path))

                # Check if the label is already assigned an ID
                if label not in label_to_id:
                    label_to_id[label] = current_id
                    current_id += 1

                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                faces.append(image)
                labels.append(label_to_id[label])  # Use the integer ID as the label

    # Convert labels to numpy array
    labels = np.array(labels)

    # Train the model
    recognizer.train(faces, labels)

    # Save the trained model
    recognizer.save(trainedModelPath)

    messagebox.showinfo("Success", "Model training completed.")

# Create the GUI window
window = Tk()
window.title("Attendance System")
window.geometry("800x400")

# Set a background image
image = Image.open("background_image.png")
window_width, window_height = 800, 400
image = image.resize((window_width, window_height), Image.ANTIALIAS)
photo = ImageTk.PhotoImage(image)
background_label = Label(window, image=photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Calculate the center coordinates of the window
center_x = window_width // 2
center_y = window_height // 2

# Create a label frame for the progress and ETA
progress_frame = LabelFrame(window, text="Training Progress", padx=10, pady=10, bg="white")
progress_frame.place(x=center_x, y=center_y, anchor="center")

# Create labels to display progress and ETA
progress_label = Label(progress_frame, text="Progress: 0%", font=("Arial", 14), bg="white")
progress_label.pack(pady=5)

# Create a button to trigger model training
train_button = Button(progress_frame, text="Train Model", font=("Arial", 14), command=train_model)
train_button.pack(pady=10)

# Run the GUI
window.mainloop()

