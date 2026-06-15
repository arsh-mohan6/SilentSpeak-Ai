import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw")

csv_files = list(DATA_PATH.glob("*/*.csv"))

print(f"Found {len(csv_files)} CSV files")

all_data = []

for file in csv_files:
    df = pd.read_csv(file)

    print(f"{file.name} -> {df.shape}")

    all_data.append(df)

merged_df = pd.concat(all_data, ignore_index=True)

print("\nMerged Shape:", merged_df.shape)

OUTPUT_PATH = Path("data/processed")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

merged_df.to_csv(
    OUTPUT_PATH / "merged_dataset.csv",
    index=False
)

print("\nDataset saved successfully!")