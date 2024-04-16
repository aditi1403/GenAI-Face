import cv2
import dlib
import math
import os
import random
# import openai
# from openai import OpenAI
import torch
from diffusers import StableDiffusionPipeline
# from PIL import Image
import numpy as np

# # Initialize global variable
global detected_face_shape
detected_face_shape = None

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load facial landmark detection model
p = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(p)

### FACE SHAPE DETECTION PART ###

# Utility functions to calculate distance and ratios:
def euclidean_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def forehead_width(landmarks):
    return euclidean_distance((landmarks.part(2).x, landmarks.part(2).y), (landmarks.part(21).x, landmarks.part(21).y))

def face_width(landmarks):
    return max(jawline_length(landmarks), forehead_width(landmarks))

def face_height(landmarks):
    return euclidean_distance((landmarks.part(1).x, landmarks.part(1).y), (landmarks.part(15).x, landmarks.part(15).y))

def jawline_length(landmarks):
    return euclidean_distance((landmarks.part(0).x, landmarks.part(0).y), (landmarks.part(16).x, landmarks.part(16).y))

def cheekbone_length(landmarks):
    return euclidean_distance((landmarks.part(10).x, landmarks.part(10).y), (landmarks.part(15).x, landmarks.part(15).y))

def ear_length(landmarks):
    return euclidean_distance((landmarks.part(4).x, landmarks.part(4).y), (landmarks.part(18).x, landmarks.part(18).y))

def interocular_distance(landmarks):
    return euclidean_distance((landmarks.part(36).x, landmarks.part(36).y), (landmarks.part(41).x, landmarks.part(41).y))

def eye_width(landmarks):
    return interocular_distance(landmarks) * 0.5

def jaw_width_to_height_ratio(landmarks):return jawline_length(landmarks) / face_height(landmarks)

def cheekbones_to_ear_ratio(landmarks):
    return cheekbone_length(landmarks) / ear_length(landmarks)


# Calculate_shape function : landmarks
def calculate_shape(landmarks):
    global detected_face_shape

    if 0.85 <= jaw_width_to_height_ratio(landmarks) <= 1.15 and 0.95 <= cheekbones_to_ear_ratio(landmarks) <= 1.05:
        detected_face_shape = 'Round'
    
    elif 0.95 <= jaw_width_to_height_ratio(landmarks) <= 1.05 and abs(face_width(landmarks) - face_height(landmarks)) < 0.1 * face_height(landmarks):
        detected_face_shape = 'Square'
    
    elif 1.25 <= cheekbones_to_ear_ratio(landmarks) <= 1.35 and forehead_width(landmarks) / face_width(landmarks) < 0.45 and jawline_length(landmarks) / face_width(landmarks) > 0.55:
        detected_face_shape = 'Diamond'
    
    else:
        detected_face_shape = 'Unknown'

# detect_faces

def detect_faces(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray_frame[y:y+h, x:x+w]
        landmarks = predictor(face, dlib.rectangle(0, 0, w, h))
        shape = calculate_shape(landmarks)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, detected_face_shape, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame

def landmarks_for_face(face):
    return predictor(face, dlib.rectangle(left=0, top=0, right=face.shape[1], bottom=face.shape[0]))

# Initialize camera
cap = cv2.VideoCapture(0)

# Check if the camera is opened
if not cap.isOpened():
    print("Unable to read camera feed")
    exit()

# Capture image from the camera
ret, start_image = cap.read()

if not ret:
    print("Unable to capture image from the camera")
    exit()

# Save the image in a new folder
output_folder = "face_shapes_images"
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

image_path = os.path.join(output_folder, "start_image.jpg")
cv2.imwrite(image_path, start_image)

print("Start image saved.")

# Release the camera
cap.release()

# Initialize camera again
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Process frame
    frame = detect_faces(frame)

    cv2.imshow('Face Shapes', frame)

    if detected_face_shape is not None:
        # Display the detected face shape
        cv2.putText(frame, detected_face_shape, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # print(f"Detected face shape: {detected_face_shape}")

    if cv2.waitKey(1) == ord('q'):
        break

print(f"Detected face shape: {detected_face_shape}")
print("Camera images processed.")

### JEWELS RECOMMENDATION PART ###

if detected_face_shape is not None:
    # # Define dataset path and shapes folder
    dataset_path = "C:/Users/aditi/OneDrive/Desktop/DESKTOP/VS_LANG/Python/AI/Jewels_dataset"
    d_face_shape = detected_face_shape
    shapes_folder = os.path.join(dataset_path, d_face_shape)
    # print("shapes folder",shapes_folder)

    # Check if the shape folder exists
    if not os.path.exists(shapes_folder):
        print(f"The '{d_face_shape}' folder does not exist in the dataset.")
        exit()

    jewelry_items=[]
    for images in os.listdir(shapes_folder):
    
        # check if the image ends with png or jpg or jpeg
        if (images.endswith(".png") or images.endswith(".jpg")\
            or images.endswith(".jpeg")):
            # display
            jewelry_items.append(images)

    # print("jewelery",jewelry_items)        

    # # Select a random jewelry item
    selected_jewel = random.choice(jewelry_items)
    print("selected jewel",selected_jewel)

    if not selected_jewel:
        print(f"No jewelry items available in the '{d_face_shape}/{selected_jewel}' folder.")
        exit()

        
# ### IMAGE REGENERATION PART ###

# ##### stable-diffusion-compvis/runwayml/stabilityai ##### #

generator = torch.Generator().manual_seed(1024)
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4") 
prompt = "a photograph of a woman with {d_face_shape} face shape wearing {selected_jewel} jewelery"
image = pipe(prompt, generator=generator).images[0]
image.show()


# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
