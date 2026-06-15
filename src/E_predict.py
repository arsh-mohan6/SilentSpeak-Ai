import cv2
import joblib
import mediapipe as mp
import numpy as np
from collections import deque, Counter

model = joblib.load("models/random_forest_v1.pkl")


print("Model Loaded Successfully!")

prediction_history = deque(maxlen=10)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("MediaPipe Initialized!")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

print("Webcam Started!")

cv2.namedWindow("SilentSpeak AI", cv2.WINDOW_NORMAL)
cv2.resizeWindow("SilentSpeak AI", 1280, 720)

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:
            
            
            features = []

            for lm in hand_landmarks.landmark:
                features.append(lm.x)
                features.append(lm.y)
                features.append(lm.z)

            prediction = model.predict([features])[0]

            prediction_history.append(prediction)

            prediction = Counter(prediction_history).most_common(1)[0][0]
            
            confidence = np.max(
                model.predict_proba([features])
            ) * 100
            
            
            # Black box
            cv2.rectangle(
                frame,
                (10, 10),
                (330, 100),
                (0, 0, 0),
                -1
            )

            # Prediction
            cv2.putText(
                frame,
                f"Prediction: {prediction}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                    2
            )

            # Confidence
            cv2.putText(
                frame,
                f"Confidence: {confidence:.1f}%",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape

                x_list = []
                y_list = []

                for lm in hand_landmarks.landmark:
                    x_list.append(int(lm.x * w))
                    y_list.append(int(lm.y * h))

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

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("SilentSpeak AI", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Web is sleeping now..>>")
        break
    

    if cv2.getWindowProperty("SilentSpeak AI", cv2.WND_PROP_VISIBLE) < 1:
        print("Web is sleeping now..>>")
        
        break

cap.release()
cv2.destroyAllWindows()