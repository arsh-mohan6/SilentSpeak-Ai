import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
import win32com.client

from collections import deque, Counter


# =========================================================
# LOAD MODEL AND SCALER
# =========================================================

model = joblib.load("../models/mlp_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

print("MLP Model Loaded Successfully!")
print("Scaler Loaded Successfully!")


# =========================================================
# DIGIT TO WORD
# =========================================================

digit_words = {
    0: "Zero",
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine"
}


# =========================================================
# WINDOWS TEXT-TO-SPEECH
# =========================================================

speaker = win32com.client.Dispatch("SAPI.SpVoice")

speaker.Rate = 0
speaker.Volume = 100

print("Windows Text-to-Speech Initialized!")


def speak(word):

    print("Speaking:", word)

    speaker.Speak(
        word,
        1
    )


# =========================================================
# PREDICTION CONTROL
# =========================================================

stable_prediction = None
stable_count = 0

prediction_history = deque(maxlen=10)


# =========================================================
# MEDIAPIPE INITIALIZATION
# =========================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("MediaPipe Initialized!")


# =========================================================
# START WEBCAM
# =========================================================

cap = cv2.VideoCapture(0)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


if not cap.isOpened():

    print("Could not open webcam")
    exit()


print("Webcam Started!")


# =========================================================
# CREATE WINDOW
# =========================================================

cv2.namedWindow(
    "SilentSpeak AI",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "SilentSpeak AI",
    1280,
    720
)


# =========================================================
# REAL-TIME LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # -----------------------------------------------------
    # BGR → RGB
    # -----------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb_frame)


    # =====================================================
    # HAND DETECTED
    # =====================================================

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:


            # -------------------------------------------------
            # EXTRACT 63 FEATURES
            # -------------------------------------------------

            features = []

            for lm in hand_landmarks.landmark:

                features.append(lm.x)
                features.append(lm.y)
                features.append(lm.z)


            # -------------------------------------------------
            # NUMPY ARRAY
            # -------------------------------------------------

            features = np.array(
                features
            ).reshape(1, -1)


            # -------------------------------------------------
            # FEATURE NAMES
            # -------------------------------------------------

            feature_names = [
                f"{axis}{i:02d}"
                for i in range(21)
                for axis in ["x", "y", "z"]
            ]


            # -------------------------------------------------
            # DATAFRAME
            # -------------------------------------------------

            features = pd.DataFrame(
                features,
                columns=feature_names
            )


            # -------------------------------------------------
            # SCALE FEATURES
            # -------------------------------------------------

            features_scaled = scaler.transform(
                features
            )


            # -------------------------------------------------
            # PREDICTION
            # -------------------------------------------------

            prediction = model.predict(
                features_scaled
            )[0]


            prediction_history.append(
                prediction
            )


            # -------------------------------------------------
            # SMOOTH PREDICTION
            # -------------------------------------------------

            prediction = Counter(
                prediction_history
            ).most_common(1)[0][0]


            # =================================================
            # STABILITY CHECK
            # =================================================

            if prediction == stable_prediction:

                stable_count += 1

            else:

                stable_prediction = prediction
                stable_count = 1


            # =================================================
            # SPEAK
            # =================================================

            if stable_count == 10:

                word = digit_words[prediction]

                speak(word)


            # -------------------------------------------------
            # REPEAT SAME WORD
            # -------------------------------------------------

            if (
                stable_count > 10
                and
                (stable_count - 10) % 30 == 0
            ):

                word = digit_words[prediction]

                speak(word)


            # =================================================
            # CONFIDENCE
            # =================================================

            probabilities = model.predict_proba(
                features_scaled
            )[0]

            confidence = np.max(
                probabilities
            ) * 100


            # =================================================
            # INFORMATION BOX
            # =================================================

            cv2.rectangle(
                frame,
                (10, 10),
                (360, 110),
                (0, 0, 0),
                -1
            )


            # -------------------------------------------------
            # PREDICTION TEXT
            # -------------------------------------------------

            cv2.putText(
                frame,
                f"Prediction: {prediction}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )


            # -------------------------------------------------
            # CONFIDENCE TEXT
            # -------------------------------------------------

            cv2.putText(
                frame,
                f"Confidence: {confidence:.1f}%",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )


            # =================================================
            # BOUNDING BOX
            # =================================================

            h, w, _ = frame.shape

            x_list = []
            y_list = []


            for lm in hand_landmarks.landmark:

                x_list.append(
                    int(lm.x * w)
                )

                y_list.append(
                    int(lm.y * h)
                )


            x_min = min(x_list)
            y_min = min(y_list)

            x_max = max(x_list)
            y_max = max(y_list)


            cv2.rectangle(
                frame,
                (x_min - 20, y_min - 20),
                (x_max + 20, y_max + 20),
                (0, 255, 0),
                2
            )


            # =================================================
            # DRAW LANDMARKS
            # =================================================

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )


    # =====================================================
    # NO HAND DETECTED
    # =====================================================

    else:

        prediction_history.clear()

        stable_prediction = None

        stable_count = 0


    # =====================================================
    # DISPLAY
    # =====================================================

    cv2.imshow(
        "SilentSpeak AI",
        frame
    )


    # =====================================================
    # KEYBOARD CONTROLS
    # =====================================================

    key = cv2.waitKey(1) & 0xFF


    # Q OR ESC

    if key == ord("q") or key == 27:

        print("Web is sleeping now..>>")

        break


    # =====================================================
    # WINDOW CLOSE
    # =====================================================

    if cv2.getWindowProperty(
        "SilentSpeak AI",
        cv2.WND_PROP_VISIBLE
    ) < 1:

        print("Web is sleeping now..>>")

        break


# =========================================================
# RELEASE
# =========================================================

cap.release()

cv2.destroyAllWindows()

print("Program Closed!")