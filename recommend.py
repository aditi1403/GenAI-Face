import os
import cv2
import dlib
import face
import random
import math

# Detect the face shape; store actual detected face shape
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
_, frame = cap.read()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        face = gray_frame[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]]

        p = "shape_predictor_68_face_landmarks.dat"
        predictor = dlib.shape_predictor(p)

        landmarks = predictor(face, dlib.rectangle(0, 0, face.shape[1], face.shape[0]))

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

        calculate_shape(landmarks)

        break

cap.release()


# Define dataset path and shapes folder
dataset_path = "C:/Users/aditi/OneDrive/Desktop/DESKTOP/VS_LANG/Python/AI/Jewels_dataset"
d_face_shape = detected_face_shape
shapes_folder = os.path.join(dataset_path, d_face_shape)

# Check if the shape folder exists
if not os.path.exists(shapes_folder):
    print(f"The '{d_face_shape}' folder does not exist in the dataset.")
    exit()

# Get list of all jewelry items in the shape folder
jewelry_items = [f for f in os.listdir(shapes_folder) if os.path.isdir(os.path.join(shapes_folder, f))]

# Check if there are any jewelry items in the folder
if not jewelry_items:
    print(f"No jewelry items available in the '{d_face_shape}' folder.")
    exit()

# Select a random jewelry item
selected_jewel = random.choice(jewelry_items)

# Combine paths to the selected jewelry item images
selected_jewel_images = [os.path.join(shapes_folder, selected_jewel, f) for f in os.listdir(os.path.join(shapes_folder, selected_jewel)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not selected_jewel_images:
    print(f"No jewelry items available in the '{d_face_shape}/{selected_jewel}' folder.")
    exit()

# Display the selected jewelry item
print(f"Random jewelry item: {selected_jewel_images[0]}")
frame, detected_face_shape = face.detect_faces(frame, detected_face_shape)
cv2.imshow("Random Jewelry Item", cv2.imread(selected_jewel_images[0]))

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()