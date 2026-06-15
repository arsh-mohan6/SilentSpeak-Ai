from pathlib import Path

# Project Root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Directories
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# Output File
LANDMARKS_CSV = PROCESSED_DATA_DIR / "landmarks.csv"

# Gesture Classes
GESTURE_CLASSES = [
    "Hello",
    "Bye",
    "Yes",
    "No",
    "Perfect",
    "ThankYou"
]