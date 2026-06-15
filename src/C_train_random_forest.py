import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/val.csv")

# print("Train Shape:", train_df.shape)
# print("Validation Shape:", val_df.shape)


X_train = train_df.drop(columns=["Label"])
y_train = train_df["Label"]

X_val = val_df.drop(columns=["Label"])
y_val = val_df["Label"]

# print("X_train:", X_train.shape)
# print("y_train:", y_train.shape)

# print("X_val:", X_val.shape)
# print("y_val:", y_val.shape)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training Complete!")


y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)

print(f"\nValidation Accuracy: {accuracy:.4f}")

test_df = pd.read_csv("data/processed/test.csv")

X_test = test_df.drop(columns=["Label"])
y_test = test_df["Label"]

y_pred_test = model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Test Accuracy: {test_accuracy:.4f}")

import joblib

joblib.dump(
    model,
    "models/random_forest_v1.pkl"
)