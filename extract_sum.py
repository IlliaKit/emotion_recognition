import os
import pandas as pd

BASE_DIR = "dataset"
OUTPUT_DIR = "csv_outputs"
FINAL_OUTPUT = "emotions_all.csv"

print("\n📦 Łączenie wszystkich plików CSV z folderu:", OUTPUT_DIR)

# 🔍 Znajdź wszystkie pliki CSV
csv_files = [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if f.endswith(".csv")]

if not csv_files:
    print("❌ Brak plików CSV do połączenia! Upewnij się, że folder csv_outputs istnieje.")
    exit()

# 📥 Wczytaj i połącz wszystkie dane
dfs = []
for file in csv_files:
    print(f"➡️  Wczytywanie: {os.path.basename(file)}")
    df = pd.read_csv(file)
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)

# 💾 Zapisz połączony plik
combined.to_csv(FINAL_OUTPUT, index=False, encoding="utf-8")

print(f"\n✅ Wszystkie dane połączono i zapisano do: {FINAL_OUTPUT}")

# 📊 Podsumowanie
print("\n📊 Liczba rekordów na emocję:")
print(combined["label"].value_counts())
