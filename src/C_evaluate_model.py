# C - Evaluate MLP

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# Loading Dataset

data_path = "../data/processed/merged_dataset.csv"

dataset = pd.read_csv(data_path)

print("Dataset shape:", dataset.shape)


# Separate X and y

X = dataset.drop(columns=["Single Image Frame", "Label"])
y = dataset["Label"]

print("X shape:", X.shape)
print("y shape:", y.shape)


# Same train/test split used in B

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# Load trained model and scaler

model = joblib.load("../models/mlp_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

print("\nModel and scaler loaded successfully.")


# Scale test data

X_test = scaler.transform(X_test)

print("\nTest data scaled successfully.")
print("X_test shape:", X_test.shape)


# Make predictions

y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred[:20])

print("\nActual:")
print(y_test.iloc[:20].values)


# Accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", accuracy)


# Classification Report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)