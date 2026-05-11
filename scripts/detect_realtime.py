import cv2
import pickle
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# Model Load
with open('models/sign_model.pkl', 'rb') as f:
    model = pickle.load(f)

# MediaPipe Setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = 'models/hand_landmarker.task'
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),    # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),    # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17), (17, 0)  # palm
]

signs = model.classes_

BAR_X = 10
BAR_START_Y = 100
BAR_HEIGHT = 18
BAR_GAP = 28
BAR_MAX_WIDTH = 180


def draw_confidence_bars(frame, proba):
    # dark background box for bars
    overlay = frame.copy()
    box_h = len(signs) * BAR_GAP + 20
    cv2.rectangle(overlay, (BAR_X - 5, BAR_START_Y - 25), (BAR_X + BAR_MAX_WIDTH + 70, BAR_START_Y + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    cv2.putText(frame, 'Confidence', (BAR_X, BAR_START_Y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    for i, (sign, prob) in enumerate(zip(signs, proba)):
        y = BAR_START_Y + i * BAR_GAP
        bar_w = int(prob * BAR_MAX_WIDTH)

        # color: green for top, grey for rest
        color = (0, 220, 0) if prob == max(proba) else (100, 100, 100)

        cv2.rectangle(frame, (BAR_X, y), (BAR_X + bar_w, y + BAR_HEIGHT), color, -1)
        cv2.rectangle(frame, (BAR_X, y), (BAR_X + BAR_MAX_WIDTH, y + BAR_HEIGHT), (60, 60, 60), 1)

        cv2.putText(frame, sign, (BAR_X + BAR_MAX_WIDTH + 5, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        pct = f'{prob * 100:.0f}%'
        cv2.putText(frame, pct, (BAR_X + bar_w - 28, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)


# Webcam
cap = cv2.VideoCapture(0)
print("Camera chal raha hai — Q dabao band karne ke liye!")

timestamp_ms = 0

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        timestamp_ms += 33  # approx 30 fps

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]

            # Draw landmarks
            h, w, _ = frame.shape
            for lm in landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            for connection in HAND_CONNECTIONS:
                start = landmarks[connection[0]]
                end = landmarks[connection[1]]
                start_x, start_y = int(start.x * w), int(start.y * h)
                end_x, end_y = int(end.x * w), int(end.y * h)
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

            # Features
            row = []
            for lm in landmarks:
                row.append(lm.x)
                row.append(lm.y)

            # Prediction
            proba = model.predict_proba([row])[0]
            prediction = model.classes_[np.argmax(proba)]
            confidence = max(proba) * 100

            # Confidence bars
            draw_confidence_bars(frame, proba)

            # Main label on top
            cv2.putText(frame, f'{prediction}  {confidence:.0f}%',
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 0), 3)

        cv2.imshow('Sign Language Detector', frame)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()