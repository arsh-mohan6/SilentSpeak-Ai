# G - Compare All Models

import pandas as pd
import joblib
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# Loading Dataset

data_path = "../data/processed/merged_dataset.csv"

dataset = pd.read_csv(data_path)

print("Dataset shape:", dataset.shape)


# Separate X and y

X = dataset.drop(columns=["Single Image Frame", "Label"])
y = dataset["Label"]


# Same train/test split used in all models

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# Load Models

mlp = joblib.load("../models/mlp_model.pkl")
mlp_scaler = joblib.load("../models/scaler.pkl")

svm = joblib.load("../models/svm_model.pkl")
svm_scaler = joblib.load("../models/svm_scaler.pkl")

random_forest = joblib.load("../models/random_forest_model.pkl")

knn = joblib.load("../models/knn_model.pkl")
knn_scaler = joblib.load("../models/knn_scaler.pkl")

print("\nAll models loaded successfully.")


# Store Results

results = []


# MLP

X_test_mlp = mlp_scaler.transform(X_test)

start = time.perf_counter()

y_pred_mlp = mlp.predict(X_test_mlp)

mlp_time = time.perf_counter() - start

results.append([
    "MLP",
    accuracy_score(y_test, y_pred_mlp),
    precision_score(y_test, y_pred_mlp, average="weighted"),
    recall_score(y_test, y_pred_mlp, average="weighted"),
    f1_score(y_test, y_pred_mlp, average="weighted"),
    mlp_time
])


# SVM

X_test_svm = svm_scaler.transform(X_test)

start = time.perf_counter()

y_pred_svm = svm.predict(X_test_svm)

svm_time = time.perf_counter() - start

results.append([
    "SVM",
    accuracy_score(y_test, y_pred_svm),
    precision_score(y_test, y_pred_svm, average="weighted"),
    recall_score(y_test, y_pred_svm, average="weighted"),
    f1_score(y_test, y_pred_svm, average="weighted"),
    svm_time
])


# Random Forest

start = time.perf_counter()

y_pred_rf = random_forest.predict(X_test)

rf_time = time.perf_counter() - start

results.append([
    "Random Forest",
    accuracy_score(y_test, y_pred_rf),
    precision_score(y_test, y_pred_rf, average="weighted"),
    recall_score(y_test, y_pred_rf, average="weighted"),
    f1_score(y_test, y_pred_rf, average="weighted"),
    rf_time
])


# KNN

X_test_knn = knn_scaler.transform(X_test)

start = time.perf_counter()

y_pred_knn = knn.predict(X_test_knn)

knn_time = time.perf_counter() - start

results.append([
    "KNN",
    accuracy_score(y_test, y_pred_knn),
    precision_score(y_test, y_pred_knn, average="weighted"),
    recall_score(y_test, y_pred_knn, average="weighted"),
    f1_score(y_test, y_pred_knn, average="weighted"),
    knn_time
])


# Comparison Table

comparison = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Prediction Time (seconds)"
    ]
)

print("\n================ MODEL COMPARISON ================\n")

print(comparison.to_string(index=False))