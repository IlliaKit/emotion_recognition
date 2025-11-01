import os
import sys
import subprocess
import pandas as pd

BASE_DIR = "dataset"
OUTPUT_DIR = "csv_outputs"
FINAL_OUTPUT = "emotions_all.csv"
SCRIPT = "extract_single.py"

os.makedirs(OUTPUT_DIR, exist_ok=True)

emotions = [e for e in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, e))]
print(f"\n🔍 Znalezione emocje: {emotions}\n")

for emotion in emotions:
    print(f"\n🚀 Uruchamianie procesu dla: {emotion}")
    result = subprocess.run([sys.executable, SCRIPT, emotion])
    if result.returncode != 0:
        print(f"⚠️ Błąd podczas przetwarzania: {emotion} (kod {result.returncode})")
    else:
        print(f"✅ Zakończono {emotion}\n")

print("\n📦 Łączenie wszystkich CSV w jeden plik...")
csv_files = [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv')]

if not csv_files:
    print("❌ Brak plików CSV do połączenia!")
    sys.exit(1)

combined = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
combined.to_csv(FINAL_OUTPUT, index=False, encoding='utf-8')

print(f"\n💾 Wszystkie dane połączono i zapisano do: {FINAL_OUTPUT} ✅")
print("\n📊 Liczba rekordów na emocję:")
print(combined['label'].value_counts())
