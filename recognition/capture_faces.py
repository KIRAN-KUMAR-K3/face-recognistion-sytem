# recognition/capture_faces.py
import cv2
import face_recognition
import os
import pickle

def capture_face(name):
    video_capture = cv2.VideoCapture(0)
    print("Capturing face... Press 'q' to quit")

    while True:
        ret, frame = video_capture.read()
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    face_locations = face_recognition.face_locations(frame)
    if len(face_locations) != 1:
        print("Error: No face or multiple faces detected. Please try again.")
        return

    face_encoding = face_recognition.face_encodings(frame, face_locations)[0]

    if not os.path.exists('known_faces'):
        os.makedirs('known_faces')

    with open(f'known_faces/{name}.pkl', 'wb') as f:
        pickle.dump(face_encoding, f)

if __name__ == "__main__":
    name = input("Enter your name: ")
    capture_face(name)
