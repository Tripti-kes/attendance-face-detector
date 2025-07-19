import face_recognition
import cv2
import numpy as np
import os
import csv
from datetime import datetime   

video_capture = cv2.VideoCapture(0)

Tripti_image = face_recognition.load_image_file("members/Tripti.jpg")
Tripti_encodings = face_recognition.face_encodings(Tripti_image)[0]

Debu_image = face_recognition.load_image_file("members/Debu.jpg")
Debu_encodings = face_recognition.face_encodings(Debu_image)[0]

Kirti_image = face_recognition.load_image_file("members/Kirti.jpg")
Kirti_encodings = face_recognition.face_encodings(Kirti_image)[0]

Ribhu_image = face_recognition.load_image_file("members/Ribhu.jpg")
Ribhu_encodings = face_recognition.face_encodings(Ribhu_image)[0]

known_face_encodings = [
    Tripti_encodings,
    Debu_encodings,
    Kirti_encodings,
    Ribhu_encodings
]


known_face_names = [
    "Tripti",
    "Debu",
    "Kirti",
    "Ribhu"
]

members = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True


now = datetime.now()
current_time = now.strftime("%Y-%m-%d")

f = open(current_time + '.csv', 'w+', newline='')
inwriter = csv.writer(f)

while True:
   _,frame = video_capture.read()
   small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
   rgb_small_frame = small_frame[:, :, ::-1]
   if s:
      face_locations = face_recognition.face_locations(rgb_small_frame)
      face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
      face_names = []
      for face_encodings in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, known_face_encodings)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encodings)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                
            face_names.append(name)
            if name in known_face_names:
                if name in members:
                    members.remove(name)
                    print(name + " is present")
                    current_time = now.strftime("%H:%M:%S")
                    inwriter.writerow([name, current_time])
                    cv2.imshow("attendence system", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    video_capture.release()
                    cv2.destroyAllWindows()
                    f.close()
        
    

   
     

                
         







