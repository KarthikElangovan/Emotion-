# pip install face_recognition
import os
import cv2
import face_recognition

# Set working directory where images and cascades are located
os.chdir(r"E:\Face_recognisation-main")
print("Working directory:", os.getcwd())

# Load cascade classifiers
nose_cascade_path = 'haarcascade_mcs_nose.xml'
mouth_cascade_path = 'haarcascade_mcs_mouth.xml'
eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
smile_cascade_path = cv2.data.haarcascades + 'haarcascade_smile.xml'

# Load custom and OpenCV-provided cascades
nose_cascade = cv2.CascadeClassifier(nose_cascade_path)
mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

# List of image filenames
image_paths = ['181.jpeg', '182.jpeg', '183.jpeg', '184.jpeg', '185.jpeg', '186.jpeg', '191.jpeg']

# Process each image
for image_path in image_paths:
    image = cv2.imread(image_path)

    if image is None:
        print(f"⚠️ Could not load image: {image_path}")
        continue

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)

    for (top, right, bottom, left) in face_locations:
        # Draw face box
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, 'Face', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        face_roi = image[top:bottom, left:right]
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Eyes
        eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_roi, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
            cv2.putText(image, 'Eye', (left + ex, top + ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Nose
        noses = nose_cascade.detectMultiScale(gray_face, scaleFactor=1.3, minNeighbors=5)
        for (nx, ny, nw, nh) in noses:
            cv2.rectangle(face_roi, (nx, ny), (nx + nw, ny + nh), (0, 255, 255), 2)
            cv2.putText(image, 'Nose', (left + nx, top + ny - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            break  # Draw one nose

        # Mouth (Smile)
        smiles = smile_cascade.detectMultiScale(gray_face, scaleFactor=1.8, minNeighbors=20)
        for (sx, sy, sw, sh) in smiles:
            if sy > gray_face.shape[0] // 2:  # Only lower half of face
                cv2.rectangle(face_roi, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
                cv2.putText(image, 'Mouth', (left + sx, top + sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                break

    cv2.imshow(f'Detected Features - {image_path}', image)
    cv2.waitKey(0)

cv2.destroyAllWindows()

