import face_recognition
import cv2
import numpy as np
import os
import csv
from datetime import datetime

# Load known faces
path = "members"
images = []
names = []
photo_filenames = []

for file in os.listdir(path):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        name = os.path.splitext(file)[0]
        img_path = os.path.join(path, file)
        img = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(img)
        print(f"Loaded {name} from {file} is present")
        if len(encodings) > 0:
            images.append(encodings[0])
            names.append(name)
            photo_filenames.append(file)
        else:
            print(f"[WARNING] No face found in {file}, skipping...")

marked_present = set()
now = datetime.now()
date_str = now.strftime("%Y-%m-%d")
csv_filename = f"{date_str}.csv"

with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Time", "Photo"])

    video_capture = cv2.VideoCapture(0)
    print("Attendance system started. Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        encodings_in_frame = face_recognition.face_encodings(rgb_small_frame, face_locations)

        height, width, _ = frame.shape
        # Draw one fixed RGB (blue) square in center for guidance
        box_w, box_h = 250, 250
        top_left = (width // 2 - box_w // 2, height // 2 - box_h // 2)
        bottom_right = (width // 2 + box_w // 2, height // 2 + box_h // 2)
        cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)  # RGB color (blue)

        for (top, right, bottom, left), face_encoding in zip(face_locations, encodings_in_frame):
            # Scale face box to original frame size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw ONE green box per detected face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Check if face center is inside the blue square
            face_center_x = (left + right) // 2
            face_center_y = (top + bottom) // 2
            inside_center = (top_left[0] < face_center_x < bottom_right[0]) and (top_left[1] < face_center_y < bottom_right[1])

            name = "Unknown"
            if inside_center:
                matches = face_recognition.compare_faces(images, face_encoding)
                face_distances = face_recognition.face_distance(images, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = names[best_match_index]
                    if name not in marked_present:
                        marked_present.add(name)
                        time_str = datetime.now().strftime("%H:%M:%S")
                        writer.writerow([name, time_str, photo_filenames[best_match_index]])
                        print(f"{name} is present")

            # Show the name on the frame below the green box
            cv2.putText(frame, name, (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
print("Attendance system closed.")
