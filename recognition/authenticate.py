# recognition/authenticate.py
import cv2
import face_recognition
import os
import pickle

def load_known_faces():
    known_faces = {}
    for filename in os.listdir('known_faces'):
        if filename.endswith('.pkl'):
            with open(f'known_faces/{filename}', 'rb') as f:
                known_faces[filename[:-4]] = pickle.load(f)
    return known_faces

def authenticate():
    video_capture = cv2.VideoCapture(0)
    known_faces = load_known_faces()
    print("Authenticating... Press 'q' to quit")

    while True:
        ret, frame = video_capture.read()
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = list(known_faces.keys())[first_match_index]

            print(f"Detected: {name}")

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    authenticate()
