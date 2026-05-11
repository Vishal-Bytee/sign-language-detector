import cv2
import pickle
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # popup window ke liye

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

# Matplotlib chart setup
plt.ion()
fig, ax = plt.subplots(figsize=(5, 4))
fig.patch.set_facecolor('#1e1e1e')
fig.canvas.manager.set_window_title('Live Confidence Chart')

bars = ax.barh(signs, [0] * len(signs), color='#555555')

ax.set_xlim(0, 1)
ax.set_facecolor('#1e1e1e')
ax.tick_params(colors='white')
ax.spines['bottom'].set_color('#444')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#444')
ax.set_xlabel('Confidence', color='white')
ax.set_title('Sign Confidence', color='white', fontsize=13)
plt.tight_layout()


def update_chart(proba):
    top = np.argmax(proba)
    for i, (bar, prob) in enumerate(zip(bars, proba)):
        bar.set_width(prob)
        bar.set_color('#00e676' if i == top else '#555555')
    fig.canvas.draw()
    fig.canvas.flush_events()


# Webcam
cap = cv2.VideoCapture(0)
print("Camera chal raha hai — Q dabao band karne ke liye!")

timestamp_ms = 0
last_chart_update = 0

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

            # update chart every ~300ms
            if timestamp_ms - last_chart_update > 300:
                update_chart(proba)
                last_chart_update = timestamp_ms

            # label on webcam
            cv2.putText(frame, f'{prediction}  {confidence:.0f}%',
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 0), 3)

        cv2.imshow('Sign Language Detector', frame)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
plt.close()