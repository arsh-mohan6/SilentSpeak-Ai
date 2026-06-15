# def initialize_mediapipe():
#     pass

# def extract_landmarks(image):
#     pass

# def process_dataset():
#     pass

# def save_csv():
#     pass

# def main():
#     pass

import mediapipe as mp




def initialize_mediapipe():
    """
        Initialize MediaPipe Hands for static image processing.
    """
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.6
    )

    return hands



    
import cv2
from pathlib import Path

def load_image(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Failed to load image: {image_path}")
        return None

    return image

def extract_landmarks(image, hands):
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    hand_landmarks = results.multi_hand_landmarks[0]

    return hand_landmarks

    """
    MAIN
    """
if __name__ == "__main__":

    hands = initialize_mediapipe()

    sample_image = next(
        Path("data/raw/Hello").glob("*.jpg")
    )

    image = load_image(sample_image)

    landmarks = extract_landmarks(image, hands)

    if landmarks:
        print("Hand detected!")
        print(
            f"Number of landmarks: {len(landmarks.landmark)}"
        )
    else:
        print("No hand detected.")