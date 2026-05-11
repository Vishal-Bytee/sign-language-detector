import cv2
import pickle
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Model Load
with open('models/sign_model.pkl', 'rb') as f:
    model = pickle.load(f)

# MediaPipe Setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = 'models/hand_landmarker.task'

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),# ring
    (0, 17), (17, 18), (18, 19), (19, 20),# pinky
    (5, 9), (9, 13), (13, 17), (17, 0)    # palm
]

signs = model.classes_

DEBUG_MODE = False  #make it true to remove time restrictions during development/testing


def is_active_time():
    if DEBUG_MODE:
        return True
    now = datetime.datetime.now().time()
    start = datetime.time(18, 0)
    end = datetime.time(22, 0)
    return start <= now <= end


# Matplotlib chart
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


def run_webcam():
    video_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1
    )

    cap = cv2.VideoCapture(0)
    timestamp_ms = 0
    last_chart_update = 0

    with HandLandmarker.create_from_options(video_options) as landmarker:
        while True:
            if timestamp_ms % 5000 == 0 and not is_active_time():
                print('Time khatam (10 PM). Band ho raha hai.')
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            timestamp_ms += 33

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                h, w, _ = frame.shape

                for lm in landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                for connection in HAND_CONNECTIONS:
                    start = landmarks[connection[0]]
                    end = landmarks[connection[1]]
                    sx, sy = int(start.x * w), int(start.y * h)
                    ex, ey = int(end.x * w), int(end.y * h)
                    cv2.line(frame, (sx, sy), (ex, ey), (0, 255, 0), 2)

                row = []
                for lm in landmarks:
                    row.append(lm.x)
                    row.append(lm.y)

                proba = model.predict_proba([row])[0]
                prediction = model.classes_[np.argmax(proba)]
                confidence = max(proba) * 100

                if timestamp_ms - last_chart_update > 300:
                    update_chart(proba)
                    last_chart_update = timestamp_ms

                cv2.putText(frame, f'{prediction}  {confidence:.0f}%',
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            else:
                cv2.putText(frame, 'No hand detected',
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Sign Language Detector', frame)

            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


def run_image():
    image_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1
    )

    path = filedialog.askopenfilename(
        title='Image select karo',
        filetypes=[('Image files', '*.jpg *.jpeg *.png')]
    )
    if not path:
        return

    image = cv2.imread(path)
    if image is None:
        messagebox.showerror('Error', 'Image load nahi hui!')
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    with HandLandmarker.create_from_options(image_options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        cv2.putText(image, 'No hand detected', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Image Result', image)
        cv2.waitKey(0)
        cv2.destroyWindow('Image Result')
        return

    landmarks = result.hand_landmarks[0]
    h, w, _ = image.shape

    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    for connection in HAND_CONNECTIONS:
        start = landmarks[connection[0]]
        end = landmarks[connection[1]]
        sx, sy = int(start.x * w), int(start.y * h)
        ex, ey = int(end.x * w), int(end.y * h)
        cv2.line(image, (sx, sy), (ex, ey), (0, 255, 0), 2)

    row = []
    for lm in landmarks:
        row.append(lm.x)
        row.append(lm.y)

    proba = model.predict_proba([row])[0]
    prediction = model.classes_[np.argmax(proba)]
    confidence = max(proba) * 100

    update_chart(proba)

    cv2.putText(image, f'{prediction}  {confidence:.0f}%',
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow('Image Result', image)
    cv2.waitKey(0)
    cv2.destroyWindow('Image Result')


# GUI 
def launch_gui():
    root = tk.Tk()
    root.title('Sign Language Detector')
    root.geometry('400x300')
    root.configure(bg='#1e1e1e')
    root.resizable(False, False)

    tk.Label(root, text='Sign Language Detector',
             font=('Helvetica', 18, 'bold'),
             bg='#1e1e1e', fg='white').pack(pady=30)

    tk.Label(root, text='Mode chunao:',
             font=('Helvetica', 11),
             bg='#1e1e1e', fg='#aaaaaa').pack()

    btn_style = {
        'font': ('Helvetica', 13, 'bold'),
        'width': 20,
        'height': 2,
        'bd': 0,
        'cursor': 'hand2'
    }

    tk.Button(root, text='📷  Real-Time Webcam',
              bg='#00e676', fg='#1e1e1e',
              command=lambda: [root.withdraw(), run_webcam(), root.deiconify()],
              **btn_style).pack(pady=12)

    tk.Button(root, text='  Upload Image',
              bg='#2979ff', fg='white',
              command=lambda: run_image(),
              **btn_style).pack(pady=4)

    now = datetime.datetime.now().strftime('%I:%M %p')
    status = '🟢 System Active' if is_active_time() else '🔴 Active only 6PM - 10PM'
    tk.Label(root, text=f'{status}   |   {now}',
             font=('Helvetica', 9),
             bg='#1e1e1e', fg='#777777').pack(pady=20)

    root.mainloop()


# ── Entry Point ──
if not is_active_time():
    root = tk.Tk()
    root.withdraw()
    now = datetime.datetime.now().strftime('%I:%M %p')
    messagebox.showwarning('Access Denied',
                           f'System only work between 6:00pm to 10:00pm.\n current time: {now}')
    root.destroy()
else:
    launch_gui()

plt.close()