import cv2
import mediapipe as mp
import numpy as np
import joblib
from xgboost import XGBClassifier

# --- Wczytanie modelu i skalera ---
print("📦 Ładowanie modelu...")
model = XGBClassifier()
model.load_model("models/emotion_model.xgb")
scaler = joblib.load("models/emotion_scaler.pkl")
le = joblib.load("models/emotion_labelencoder.pkl")

# --- Ustawienia MediaPipe (tak samo jak przy ekstrakcji danych!) ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,  # ✅ TAK SAMO jak w extract_single.py
    min_detection_confidence=0.3
)

# --- Start kamery ---
cap = cv2.VideoCapture(0)
print("📸 Uruchamianie kamery... Wciśnij 'q', aby zakończyć.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Pobranie 478 punktów (x, y)
            x_coords = [lm.x for lm in face_landmarks.landmark]
            y_coords = [lm.y for lm in face_landmarks.landmark]
            features = np.array([x_coords + y_coords])

            # Dopasowanie wymiarów do skalera
            if features.shape[1] == scaler.n_features_in_:
                features_scaled = scaler.transform(features)
                emotion_num = model.predict(features_scaled)[0]
                emotion = le.inverse_transform([emotion_num])[0]

                cv2.putText(frame, f'Emocja: {emotion}', (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "⚠️ Niezgodne wymiary cech!", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Emotion Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
