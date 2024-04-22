from flask import Flask, render_template, request, jsonify
import cv2
import os
import random
import dlib
import math

app = Flask(__name__)

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load facial landmark detection model
p = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(p)

# Initialize global variable
global detected_face_shape
detected_face_shape = None

# Utility functions to calculate distance and ratios
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

def jaw_width_to_height_ratio(landmarks):
    return jawline_length(landmarks) / face_height(landmarks)

def cheekbones_to_ear_ratio(landmarks):
    return cheekbone_length(landmarks) / ear_length(landmarks)

# Calculate_shape function
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

# Detect faces
def detect_faces(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray_frame[y:y+h, x:x+w]
        landmarks = predictor(face, dlib.rectangle(0, 0, w, h))
        calculate_shape(landmarks)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, detected_face_shape, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame

# Initialize camera
cap = cv2.VideoCapture(0)

# Check if the camera is opened
if not cap.isOpened():
    print("Unable to read camera feed")
    exit()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture():
    ret, frame = cap.read()
    if not ret:
        return jsonify({'error': 'Failed to capture image'})

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for compatibility with web display
    # Process frame to detect faces and shapes
    frame_with_faces = detect_faces(frame)

    # Save the captured image temporarily
    temp_image_path = 'static/captured_image.jpg'
    cv2.imwrite(temp_image_path, cv2.cvtColor(frame_with_faces, cv2.COLOR_RGB2BGR))

    # Return detected face shape
    return jsonify({'face_shape': detected_face_shape, 'image_path': temp_image_path})

if __name__ == '__main__':
    app.run(debug=True)
