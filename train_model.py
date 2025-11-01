# ============================================================
# 🎯 EMOTION CLASSIFIER TRAINING — STABLE VERSION
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import joblib
import os

print("📥 Wczytywanie danych...")
df = pd.read_csv("emotions_all.csv")

# --- Podgląd danych ---
print("\n📊 Liczba zdjęć na emocję:")
print(df["label"].value_counts())

# --- Usunięcie pustych wartości ---
df = df.dropna()
if df.empty:
    raise ValueError("❌ Plik emotions_all.csv jest pusty lub ma tylko NaN!")

# --- Usunięcie klas zbyt małych (np. < 3 próbki) ---
min_samples = 3
class_counts = df["label"].value_counts()
valid_classes = class_counts[class_counts >= min_samples].index
df = df[df["label"].isin(valid_classes)]

if len(valid_classes) < 2:
    raise ValueError(f"❌ Za mało klas do nauki ({len(valid_classes)}). "
                     f"Upewnij się, że masz zdjęcia dla co najmniej 2 emocji!")

print("\n✅ Używane emocje:", list(valid_classes))
print(df["label"].value_counts())

# --- Przygotowanie danych ---
X = df.drop(columns=["label"])
y = df["label"]

# --- Skalowanie i kodowanie ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --- Podział danych ---
print("\n🔧 Dzielenie danych na trening i test...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- Trening modelu ---
print("\n🧠 Trening modelu XGBoost...")
model = XGBClassifier(
    objective="multi:softmax",
    num_class=len(le.classes_),
    eval_metric="mlogloss",
    tree_method="hist",  # działa i lokalnie, i w Google Colab
    random_state=42
)
model.fit(X_train, y_train)

# --- Ewaluacja ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\n🎯 Dokładność:", round(acc * 100, 2), "%")
print("\n📋 Raport klasyfikacji:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# --- Zapis modelu i skalera ---
os.makedirs("models", exist_ok=True)

model.save_model("models/emotion_model.xgb")
joblib.dump(scaler, "models/emotion_scaler.pkl")
joblib.dump(le, "models/emotion_labelencoder.pkl")

print("\n💾 Zapisano model i skalery w folderze /models ✅")
print("\n🏁 Trening zakończony pomyślnie!")
