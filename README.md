Projekt realizuje system rozpoznawania emocji na podstawie mimiki twarzy przy użyciu bibliotek MediaPipe oraz XGBoost.
Celem aplikacji jest klasyfikacja emocji takich jak: angry, happy, sad i surprised na podstawie punktów charakterystycznych (landmarków) wykrywanych na twarzy.

Struktura projektu

emotion_recognition/
│
├── dataset/ # Zbiór danych – podfoldery z obrazami dla każdej emocji
│ ├── angry/
│ ├── happy/
│ ├── sad/
│ └── surprised/
│
├── csv_outputs/ # Folder, w którym zapisywane są pliki CSV po ekstrakcji punktów twarzy
│
├── extract_single.py # Ekstrakcja punktów twarzy dla pojedynczej emocji
├── extract_all.py # Automatyczne uruchamianie ekstrakcji dla wszystkich emocji
├── extract_sum.py # Łączenie wszystkich plików CSV w jeden zbiorczy plik
├── train_model.py # Trening modelu XGBoost na przygotowanych danych
├── main.py # Uruchomienie rozpoznawania emocji z kamery w czasie rzeczywistym
│
└── models/ # Folder z zapisanym modelem i skalami
├── emotion_model.xgb
├── emotion_scaler.pkl
└── emotion_labelencoder.pkl

Wymagania

Aby uruchomić projekt, należy zainstalować poniższe biblioteki:

pip install opencv-python mediapipe pandas scikit-learn xgboost joblib tqdm

Projekt działa w środowisku Python 3.10 lub nowszym.

Instrukcja działania
1. Ekstrakcja punktów twarzy

Na początku należy przygotować dane z folderu dataset/.
Proces ekstrakcji polega na wykrywaniu punktów twarzy dla każdej emocji.

Uruchomienie ekstrakcji dla wszystkich emocji:
python extract_all.py

Skrypt przetworzy wszystkie podfoldery w dataset/ i zapisze wyniki do csv_outputs/.

Połączenie wszystkich wyników w jeden plik:
python extract_sum.py

Po wykonaniu powstanie plik emotions_all.csv

2. Trenowanie modelu

Na podstawie wygenerowanego pliku emotions_all.csv trenujemy model XGBoost:

python train_model.py

Po zakończeniu trenowania w folderze models/ zostaną zapisane:

emotion_model.xgb – wytrenowany model klasyfikacji emocji

emotion_scaler.pkl – skaler do normalizacji danych wejściowych

emotion_labelencoder.pkl – enkoder etykiet emocji

3. Rozpoznawanie emocji z kamery

Aby uruchomić rozpoznawanie emocji w czasie rzeczywistym, należy uruchomić:

python main.py

Program otworzy okno kamery, wykryje do trzech twarzy jednocześnie i przypisze do każdej z nich emocję.
Po prawej stronie obrazu znajduje się panel, który pokazuje numer twarzy i rozpoznaną emocję.
Działanie programu można zakończyć, naciskając klawisz q.

Uwagi

Różnice w budowie twarzy (np. wysokość brwi, kształt oczu czy ust) mogą wpływać na dokładność rozpoznawania emocji.

Oświetlenie i pozycja twarzy mają duże znaczenie dla skuteczności modelu.

W przypadku dodania nowych zdjęć zaleca się ponowne przetrenowanie modelu.

Wykrywanie twarzy odbywa się w oparciu o punkty z MediaPipe (468–478 landmarków na twarz).
