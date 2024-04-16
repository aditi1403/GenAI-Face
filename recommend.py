import os
import face
import random

### JEWELS RECOMMENDATION PART ###

if face.detected_face_shape is not None:
    # # Define dataset path and shapes folder
    dataset_path = "C:/Users/aditi/OneDrive/Desktop/DESKTOP/VS_LANG/Python/AI/Jewels_dataset"
    d_face_shape = face.detected_face_shape
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
