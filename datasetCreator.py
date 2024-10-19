import cv2
import os

# Initialize the camera and create a directory to store the dataset
camera = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_id = input('\n Enter user ID and press <return>: ')

print("\n [INFO] Initializing face capture. Look at the camera and wait...")
count = 0

while True:
    ret, img = camera.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        count += 1
        # Save the captured image into the dataset folder
        cv2.imwrite(f"dataSet/User.{face_id}.{count}.jpg", gray[y:y + h, x:x + w])
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    cv2.imshow('image', img)

    # Break after 30 images
    if count >= 30:
        break

    # Press 'q' to stop capturing
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# Release the camera
camera.release()
cv2.destroyAllWindows()

print("\n [INFO] Dataset creation complete.")
