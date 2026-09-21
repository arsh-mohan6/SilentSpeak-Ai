import pandas as pd
import os

data_path = "../data/raw"

data = []

for label in range(10):
    folder = os.path.join(data_path, str(label))

    for file in os.listdir(folder):
        if file.endswith(".csv"):
            file_path = os.path.join(folder, file)

            df = pd.read_csv(file_path)
            data.append(df)

dataset = pd.concat(data, ignore_index=True)

print("Dataset shape:", dataset.shape)
print(dataset.head())

print("\nClass distribution:")
print(dataset["Label"].value_counts().sort_index())

print("\nMissing values:")
print(dataset.isnull().sum().sum())#0
print("\nDuplicate rows:", dataset.duplicated().sum())#0

os.makedirs("../data/processed", exist_ok=True)

dataset.to_csv("../data/processed/merged_dataset.csv", index=False)

print("\nMerged dataset saved successfully.")

features = dataset.drop(columns=["Single Image Frame", "Label"])

print(features.describe().T)