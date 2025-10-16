# scripts/validate_ms_dataset.py
import pandas as pd
from pathlib import Path

p = Path("data/raw/MS_dataset.csv")
df = pd.read_csv(p)

# Attese da mmc2.txt: colonna 0 = Name, 1..600 = spettro, ultima = Label (0..6)
n_cols = df.shape[1]
assert n_cols in (602, 603), f"Colonne inaspettate: {n_cols}"

name_col = df.columns[0]
label_col = df.columns[-1]
spec_cols = df.columns[1:-1]

# Tipi base
assert df[name_col].dtype == object, "La prima colonna deve essere Name (stringa)"
assert pd.api.types.is_integer_dtype(df[label_col]), "L'ultima colonna deve essere Label (int)"

# Range label
assert set(df[label_col].unique()).issubset(set(range(7))), "Label fuori range 0..6"

# Spettri numerici e senza NaN
assert df[spec_cols].apply(pd.to_numeric, errors="coerce").notna().all().all(), "Spettri con valori non numerici"
assert df[spec_cols].isna().sum().sum() == 0, "Spettri con NaN"

# Crea un sample piccolo, bilanciato per test rapidi
sample = (
    df.groupby(label_col, group_keys=False)
      .apply(lambda g: g.sample(min(len(g), 20), random_state=42))
      .reset_index(drop=True)
)
out = Path("data/processed/MS_dataset_sample.csv")
out.parent.mkdir(parents=True, exist_ok=True)
sample.to_csv(out, index=False)
print("OK ✓  Dataset valido. Sample creato in", out)
