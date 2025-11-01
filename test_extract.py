import pandas as pd

for name in ["angry", "happy", "sad", "surprised"]:
    f = f"csv_outputs/emotions_{name}.csv"
    df = pd.read_csv(f)
    print(name, "→", df.shape)