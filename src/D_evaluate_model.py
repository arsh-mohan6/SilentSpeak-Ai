import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

model = joblib.load("models/random_forest_v1.pkl")

print("Model Loaded Successfully!")

test_df = pd.read_csv("data/processed/test.csv")

X_test = test_df.drop(columns=["Label"])
y_test = test_df["Label"]

print("Test Shape:", X_test.shape)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))