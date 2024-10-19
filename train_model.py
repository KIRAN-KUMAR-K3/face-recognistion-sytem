import os
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Path to the dataset
data_path = 'dataSet/'

# Prepare training data
def prepare_data():
    face_samples = []
    ids = []
    image_paths = [os.path.join(data_path, f) for f in os.listdir(data_path)]
    
    for image_path in image_paths:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img_numpy = np.array(img, 'uint8')
        id = int(os.path.split(image_path)[-1].split(".")[1])
        
        faces = face_detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)
    
    return face_samples, ids

# Define a CNN model for face recognition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification (present/not present)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the dataset and train the model
face_samples, ids = prepare_data()
x_train = np.array(face_samples).reshape(-1, 64, 64, 1)  # Reshape for CNN input
y_train = np.array(ids)

model.fit(x_train, y_train, epochs=10, batch_size=32)

# Save the model
model.save('face_recognition_model.h5')
print("\n [INFO] Model training complete.")
