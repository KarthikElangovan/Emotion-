import cv2
import face_recognition
import os

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Haar Cascade paths (use OpenCV built-ins for eyes; custom XMLs should be in your working directory)
nose_cascade_path = 'haarcascade_mcs_nose.xml'      # Make sure this file exists
mouth_cascade_path = 'haarcascade_mcs_mouth.xml'    # Make sure this file exists

# Load classifiers
nose_cascade = cv2.CascadeClassifier(nose_cascade_path)
mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, 'Face', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        face_roi = frame[top:bottom, left:right]
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Eyes
        eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_roi, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
            cv2.putText(frame, 'Eye', (left + ex, top + ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Nose
        nose = nose_cascade.detectMultiScale(gray_face, scaleFactor=1.3, minNeighbors=5)
        for (nx, ny, nw, nh) in nose:
            cv2.rectangle(face_roi, (nx, ny), (nx + nw, ny + nh), (0, 255, 255), 2)
            cv2.putText(frame, 'Nose', (left + nx, top + ny - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Mouth
        mouth = mouth_cascade.detectMultiScale(gray_face, scaleFactor=1.8, minNeighbors=20)
        for (mx, my, mw, mh) in mouth:
            if my > gray_face.shape[0] // 2:  # Only lower part of face
                cv2.rectangle(face_roi, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
                cv2.putText(frame, 'Mouth', (left + mx, top + my - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                break  # Only one mouth

    cv2.imshow('Facial Features Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
