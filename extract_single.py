import cv2
import mediapipe as mp
import os
import csv
import sys
from tqdm import tqdm

BASE_DIR = "dataset"
OUTPUT_DIR = "csv_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADER = [f'x{i}' for i in range(478)] + [f'y{i}' for i in range(478)] + ['label']

# 🧩 Sprawdź, czy przekazano emocję jako argument
if len(sys.argv) < 2:
    print("❌ Nie podano emocji! Użycie: python extract_single.py <emotion>")
    sys.exit(1)

emotion = sys.argv[1]
folder = os.path.join(BASE_DIR, emotion)
images = [img for img in os.listdir(folder) if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
csv_path = os.path.join(OUTPUT_DIR, f"emotions_{emotion}.csv")

print(f"\n📂 Przetwarzanie emocji: {emotion} ({len(images)} zdjęć)")

processed, failed = 0, 0
mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,  # UŻYJ TEGO SAMEGO USTAWIENIA CO W main.py
    min_detection_confidence=0.3
) as face_mesh, open(csv_path, 'w', newline='', encoding='utf-8') as f:

    writer = csv.writer(f)
    writer.writerow(HEADER)

    for img_name in tqdm(images, desc=emotion, ncols=80):
        img_path = os.path.join(folder, img_name)
        image = cv2.imread(img_path)
        if image is None:
            failed += 1
            continue

        image = cv2.resize(image, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                x_coords = [lm.x for lm in face_landmarks.landmark]
                y_coords = [lm.y for lm in face_landmarks.landmark]
                writer.writerow(x_coords + y_coords + [emotion])
                processed += 1
        else:
            failed += 1

print(f"✅ {emotion}: {processed}/{len(images)} OK, {failed} błędnych ❌")
