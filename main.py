import cv2
import mediapipe as mp
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)


from xgboost import XGBClassifier

# --- Wczytanie modelu i skalera ---
print("📦 Ładowanie modelu...")
model = XGBClassifier()
model.load_model("models/emotion_model.xgb")
scaler = joblib.load("models/emotion_scaler.pkl")
le = joblib.load("models/emotion_labelencoder.pkl")

# --- MediaPipe (dokładnie jak przy ekstrakcji) ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=3,              
    refine_landmarks=True,
    min_detection_confidence=0.3
)

# --- Uruchomienie kamery ---
cap = cv2.VideoCapture(0)
print("📸 Uruchamianie kamery... Wciśnij 'q', aby zakończyć.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    # --- Panel boczny z emocjami ---
    sidebar_width = 250
    sidebar = np.zeros((h, sidebar_width, 3), dtype=np.uint8) + 40

    face_count = 0
    emotions_list = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_count += 1

            # Pobranie punktów (x, y)
            x_coords = [lm.x for lm in face_landmarks.landmark]
            y_coords = [lm.y for lm in face_landmarks.landmark]
            features = np.array([x_coords + y_coords])

            # Dopasowanie liczby cech
            if features.shape[1] == scaler.n_features_in_:
                features_scaled = scaler.transform(features)
                emotion_num = model.predict(features_scaled)[0]
                emotion = le.inverse_transform([emotion_num])[0]
            else:
                emotion = "unknown"

            # Przekształcenie współrzędnych do pikseli (Pobranie współrzędnych twarzy)
            x_coords_px = [int(lm.x * w) for lm in face_landmarks.landmark]
            y_coords_px = [int(lm.y * h) for lm in face_landmarks.landmark]
            x_min, x_max = min(x_coords_px), max(x_coords_px)
            y_min, y_max = min(y_coords_px), max(y_coords_px)

            # Kolory emocji
            colors = {
                'happy': (0, 255, 0),
                'sad': (255, 0, 0),
                'angry': (0, 0, 255),
                'surprised': (0, 255, 255),
                'neutral': (200, 200, 200),
                'unknown': (128, 128, 128)
            }
            color = colors.get(emotion, (255, 255, 255))

            # Ramka wokół twarzy
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, f'{emotion}', (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Lista emocji do panelu bocznego
            emotions_list.append((face_count, emotion))

    # --- Panel po prawej stronie ---
    cv2.putText(sidebar, "Detected faces:", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for i, (face_num, emo) in enumerate(emotions_list):
        cv2.putText(sidebar, f"{face_num} face: {emo}", (10, 80 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Połączenie obrazu i panelu
    combined_frame = np.hstack((frame, sidebar))

    cv2.imshow("Emotion Recognition", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
