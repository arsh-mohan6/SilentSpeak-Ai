# D - Train SVM

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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


# Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

print("\nData scaled successfully.")


# Create SVM Model

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

print("\nTraining SVM...")

model.fit(X_train, y_train)

print("Training completed.")


# Save SVM model and scaler

joblib.dump(model, "../models/svm_model.pkl")
joblib.dump(scaler, "../models/svm_scaler.pkl")

print("\nSVM model saved successfully.")
print("SVM scaler saved successfully.")