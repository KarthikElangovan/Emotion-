import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array

# ========== Configuration ==========
MODEL_PATH = "model.h5"
CASCADE_PATH = "haarcascade_frontalface_default.xml"
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ========== Load Model ==========
try:
    emotion_model = load_model(MODEL_PATH)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# ========== Load Haar Cascade ==========
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + CASCADE_PATH)
if face_classifier.empty():
    print("❌ Error loading Haar cascade.")
    exit()

# ========== Start Webcam ==========
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error accessing webcam.")
    exit()

# ========== Main Loop ==========
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum(roi_gray) > 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)  # Shape becomes (1, 48, 48, 1)

            try:
                prediction = emotion_model.predict(roi, verbose=0)[0]
                label = EMOTION_LABELS[np.argmax(prediction)]
            except Exception as e:
                label = "Error"
                print("Prediction error:", e)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Face', (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ========== Cleanup ==========
cap.release()
cv2.destroyAllWindows()
