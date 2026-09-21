# 🤟 SilentSpeak AI

<p align="center">
  <img src="results/model_accuracy_comparison.png" width="700">
</p>

<p align="center">
  <b>Real-Time Hand Gesture Recognition & Speech System</b>
</p>

<p align="center">
  Machine Learning • MediaPipe • OpenCV • Real-Time Computer Vision • Text-to-Speech
</p>

---

## 📌 Overview

**SilentSpeak AI** is a real-time machine learning-based hand gesture recognition system that detects hand gestures through a webcam and converts the recognized gestures into spoken words.

The system uses **MediaPipe Hands** to detect 21 hand landmarks. Since every landmark contains **X, Y, and Z coordinates**, each detected hand is represented by:

**21 landmarks × 3 coordinates = 63 features**

These features are scaled and passed to trained machine learning models for classification.

The final real-time system uses an **MLP (Multi-Layer Perceptron)** classifier to recognize digits from **0–9** and converts the predicted digit into speech using **Text-to-Speech**.

---

## ✨ Key Features

- 🎥 Real-time webcam-based gesture recognition
- ✋ MediaPipe hand landmark detection
- 📊 63-feature extraction from each hand
- 🤖 Multiple machine learning algorithms
- 🧠 MLP-based real-time classification
- 📈 Model performance comparison
- 🔄 Prediction smoothing for stable results
- 📊 Real-time prediction confidence
- 🟩 Hand bounding-box visualization
- 🔊 Text-to-Speech output
- 💾 Saved trained model for inference
- ⌨️ Q / ESC controls for closing the application

---

## 🧠 Machine Learning Models

The project implements and compares four machine learning algorithms:

- **Multi-Layer Perceptron (MLP)**
- **Support Vector Machine (SVM)**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**

The **MLP model** is used for the final real-time webcam application.

---

## 🔄 System Workflow

```text
Webcam Input
      ↓
MediaPipe Hands
      ↓
21 Hand Landmarks
      ↓
63 Features (X, Y, Z)
      ↓
StandardScaler
      ↓
MLP Classifier
      ↓
Gesture Prediction
      ↓
Prediction Smoothing
      ↓
Digit → Word Conversion
      ↓
Text-to-Speech
```

📊 Dataset

The processed dataset contains 5,000 samples.

Property	Value
Total Samples	5,000
Features	63
Classes	10
Target Classes	0–9
Features per Hand	21 × 3
Feature Representation

Each hand is represented using 21 MediaPipe landmarks:
📊 Dataset

The processed dataset contains 5,000 samples.

Property	Value
Total Samples	5,000
Features	63
Classes	10
Target Classes	0–9
Features per Hand	21 × 3
Feature Representation

Each hand is represented using 21 MediaPipe landmarks:


### PART 2/4

```markdown
---

## 📈 Model Performance

The models were evaluated using the same processed dataset and held-out test split.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| **MLP** | **100%** | **100%** | **100%** | **100%** |
| **SVM** | **100%** | **100%** | **100%** | **100%** |
| Random Forest | 99.9% | 99.90% | 99.9% | 99.9% |
| **KNN** | **100%** | **100%** | **100%** | **100%** |

> **Note:** These metrics represent performance on the held-out test dataset. They should not be interpreted as real-world webcam accuracy. Actual performance can vary depending on lighting, camera quality, hand position, background, and gesture execution.

---
```
## 🎥 Real-Time Recognition

The trained MLP model is loaded directly during webcam execution.

The system does **not retrain the model every time the webcam is opened**.

Saved files:

```text
models/
├── mlp_model.pkl
└── scaler.pkl
```
```text
SILENTSPEAK-AI/
│
├── data/
│   ├── raw/
│   │
│   └── processed/
│       └── merged_dataset.csv
│
├── models/
│   ├── mlp_model.pkl
│   └── scaler.pkl
│
├── results/
│   └── model_accuracy_comparison.png
│
├── src/
│   ├── B_mlp_training.py
│   ├── C_mlp_evaluation.py
│   ├── D_svm.py
│   ├── E_random_forest.py
│   ├── F_knn.py
│   ├── G_model_comparison.py
│   └── H_realtime_webcam.py
│
├── report/
│
├── requirements.txt
│
└── README.md
```

Move into the project directory:

cd SilentSpeak-AI

Create a virtual environment:

python -m venv venv

Activate the environment on Windows:

venv\Scripts\activate

Install the required dependencies:

pip install -r requirements.txt


### PART 4/4

```markdown
---

## 🎯 Current Recognition Scope

The current system recognizes **digit-based hand gestures from 0 to 9**.

```text
0  1  2  3  4
5  6  7  8  9
```
The current implementation focuses on static hand gestures rather than continuous sign-language sentences.

🚀 Future Improvements
🔤 Alphabet and sign-language gesture recognition
👐 Multi-hand gesture recognition
📝 Continuous gesture-to-sentence conversion
🧠 Deep learning-based gesture recognition
🌐 Web-based user interface
🌍 Multilingual speech output
📱 Mobile deployment
🎯 Larger and more diverse datasets
📊 Dedicated real-world webcam evaluation dataset
⚠️ Performance Consideration

The reported 100% test accuracy is based on the current held-out dataset evaluation.

Real-world webcam performance is a separate measurement because webcam input introduces factors such as:

Different lighting conditions
Background variations
Camera resolution
Hand orientation
Distance from camera
Gesture execution differences

Therefore, test-set accuracy and real-world webcam performance should be reported separately.

📌 Project Highlights
✔ 5,000 Dataset Samples
✔ 63 Hand Landmark Features
✔ 10 Gesture Classes
✔ 4 Machine Learning Algorithms
✔ Real-Time Webcam Recognition
✔ MediaPipe Hand Landmark Detection
✔ MLP-Based Real-Time Prediction
✔ Prediction Smoothing
✔ Confidence Display
✔ Text-to-Speech Output
✔ Saved Model & Scaler
✔ Real-Time Inference Without Retraining
👨‍💻 Author

Arsh Mohan Nishant

B.Tech CSE
KIIT University

🙏 Acknowledgements

This project uses the following open-source technologies:

MediaPipe
OpenCV
Scikit-learn
NumPy
Pandas
Joblib
pyttsx3
