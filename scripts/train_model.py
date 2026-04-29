import os
import json
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


IMAGE_PATH = 'collected_images'
ANNOTATION_PATH = 'annotations'
MODEL_SAVE_PATH = 'models/sign_model.pkl'

# ── MediaPipe Setup ──
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = 'models/hand_landmarker.task'
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

# Data Collections 
data = []
labels = []

signs = ['Hello', 'ThankYou', 'Yes', 'No', 'ILoveYou', 'Please', 'Sorry', 'eat','drink']

print("Data load ho raha hai...")

with HandLandmarker.create_from_options(options) as landmarker:
    for sign in signs:
        img_folder = os.path.join(IMAGE_PATH, sign)
        ann_folder = ANNOTATION_PATH

        for img_file in os.listdir(img_folder):
            if not img_file.endswith('.jpg'):
                continue

            

            # Image load karo
            img_path = os.path.join(img_folder, img_file)
            image = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Create mp.Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            result = landmarker.detect(mp_image)

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                row = []
                for lm in landmarks:
                    row.append(lm.x)
                    row.append(lm.y)
                data.append(row)
                labels.append(sign)

print(f"Total samples: {len(data)}")

if len(data) == 0:
    print("no data and check  annotations check.")
    exit()

# Train/Test Split 
X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model Train 
print(" model is training ")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#  Accuracy Check 
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")

# Model Save
os.makedirs('models', exist_ok=True)
with open(MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump(model, f)

print(f" Model save : {MODEL_SAVE_PATH}")
print(" Training complete!")