import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("data/processed/merged_dataset.csv")

print("Dataset Shape:", df.shape)

# Features
X = df.drop(columns=["Single Image Frame", "Label"])

# Target
y = df["Label"]

print("X Shape:", X.shape)
print("y Shape:", y.shape)


# First split: Train (70%) and Temp (30%)

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Second split: Temp -> Validation (15%) + Test (15%)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nTrain:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)


# print("\nTrain Distribution:")
# print(y_train.value_counts().sort_index())

# print("\nValidation Distribution:")
# print(y_val.value_counts().sort_index())

# print("\nTest Distribution:")
# print(y_test.value_counts().sort_index())

train_df = X_train.copy()
train_df["Label"] = y_train

val_df = X_val.copy()
val_df["Label"] = y_val

test_df = X_test.copy()
test_df["Label"] = y_test

train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/val.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("\nTrain/Val/Test datasets saved!")