import cv2
import face_recognition
import mediapipe as mp
import pickle
import os
import numpy as np
import threading

ENCODINGS_FILE = "face_encodings.pkl"
RECOGNITION_INTERVAL = 5

class FaceRecognizer:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.load_encodings()

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

        self.frame_count = 0
        self.face_locations = []
        self.face_names = []
        self.lock = threading.Lock()

    def load_encodings(self):
        if os.path.exists(ENCODINGS_FILE):
            with open(ENCODINGS_FILE, 'rb') as f:
                data = pickle.load(f)
                self.known_encodings = data['encodings']
                self.known_names = data['names']
            print(f"[INFO] Loaded {len(self.known_names)} known faces.")

    def save_encodings(self):
        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump({'encodings': self.known_encodings, 'names': self.known_names}, f)

    def detect_faces(self, rgb, w, h):
        results = self.face_detector.process(rgb)
        locations = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                xmin = int(bbox.xmin * w)
                ymin = int(bbox.ymin * h)
                box_w = int(bbox.width * w)
                box_h = int(bbox.height * h)
                top = max(ymin, 0)
                right = min(xmin + box_w, w)
                bottom = min(ymin + box_h, h)
                left = max(xmin, 0)
                locations.append((top, right, bottom, left))
        return locations

    def recognition_worker(self, rgb_frame, locations):
        names = []
        encodings = face_recognition.face_encodings(rgb_frame, locations)
        for encoding in encodings:
            matches = face_recognition.compare_faces(self.known_encodings, encoding)
            name = "Unknown"
            if True in matches:
                best_match_index = matches.index(True)
                name = self.known_names[best_match_index]
            names.append(name)
        with self.lock:
            self.face_names = names

    def add_face(self):
        cap = cv2.VideoCapture(0)
        print("[INFO] Press 'a' to capture and save face, or 'q' to quit.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            locations = self.detect_faces(rgb, w, h)

            for (top, right, bottom, left) in locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            cv2.imshow("Add Face", frame)
            key = cv2.waitKey(1)
            if key == ord('a') and len(locations) == 1:
                encoding = face_recognition.face_encodings(rgb, [locations[0]])[0]
                name = input("Enter name: ").strip()
                self.known_encodings.append(encoding)
                self.known_names.append(name)
                self.save_encodings()
                print(f"[INFO] Added '{name}' to database.")
            elif key == ord('a'):
                print("[WARN] Ensure only one face is visible.")
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def recognize_faces(self):
        cap = cv2.VideoCapture(1) # 0 for defualt cam
        print("[INFO] Starting recognition. Press 'q' to quit.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            locations = self.detect_faces(rgb, w, h)

            if self.frame_count % RECOGNITION_INTERVAL == 0 and locations:
                threading.Thread(target=self.recognition_worker, args=(rgb.copy(), locations.copy()), daemon=True).start()
                self.face_locations = locations

            with self.lock:
                for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                    text_y = top - 10 if top - 10 > 10 else top + 10
                    cv2.putText(frame, name, (left, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            self.frame_count += 1

        cap.release()
        cv2.destroyAllWindows()


def main():
    system = FaceRecognizer()
    while True:
        print("\n=== Face Recognition System ===")
        print("1. Add new face")
        print("2. Run recognition")
        print("3. Exit")
        choice = input("Select option (1/2/3): ").strip()
        if choice == '1':
            system.add_face()
        elif choice == '2':
            system.recognize_faces()
        elif choice == '3':
            break
        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()
