# E - Train Random Forest

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Loading Dataset

data_path = "../data/processed/merged_dataset.csv"

dataset = pd.read_csv(data_path)

print("Dataset shape:", dataset.shape)


# Separate X and y

X = dataset.drop(columns=["Single Image Frame", "Label"])
y = dataset["Label"]

print("X shape:", X.shape)
print("y shape:", y.shape)


# Split train/test dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)


# Create Random Forest Model

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training completed.")


# Save Random Forest model

joblib.dump(model, "../models/random_forest_model.pkl")

print("\nRandom Forest model saved successfully.")