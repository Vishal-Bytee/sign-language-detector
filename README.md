# Sign Language Detector

A real-time sign language detection system using **MediaPipe** for hand landmark detection and **Random Forest** for gesture classification. Built with Python and OpenCV.



## Demo

> Point your hand at the webcam and the model will detect your sign in real time!



##  Supported Signs

| Sign | Sign | Sign |
|------|------|------|
| Hello | Thank You | Yes |
| No | I Love You | Please |
| Sorry | Eat | Drink |


##  Project Structure
```
sign_language_detector/
─ collected_images/       # Training images (one folder per sign)
─ models/
    ─ hand_landmarker.task   # MediaPipe hand model
    ─ sign_model.pkl         # Trained classifier (generated after training)
─ scripts/
   ── collect_images.py      # Capture training images from webcam
   ── train_model.py         # Extract landmarks & train the model
   ── detect_realtime.py     # Live webcam detection
─ .gitignore
─ requirements.txt
─ README.md
```
 




##  Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Vishal-Bytee/sign-language-detector.git
cd sign-language-detector
```

### 2. Create a virtual environment
```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows (PowerShell):**
```bash
.venv\Scripts\Activate.ps1
```

**Mac/Linux:**
```bash
source .venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Download MediaPipe Hand Landmarker Model
Download `hand_landmarker.task` from the [MediaPipe official site](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) and place it inside the `models/` folder.



##  How to Use

### Step 1 — Collect Training Images
```bash
python scripts/collect_images.py
```
- Shows each sign name on screen
- Automatically captures **50 images per sign**
- Saves them in `collected_images/<sign_name>/`

### Step 2 — Train the Model
```bash
python scripts/train_model.py
```
- Extracts hand landmarks using MediaPipe
- Trains a **Random Forest Classifier**
- Saves the model to `models/sign_model.pkl`
- Prints accuracy after training

### Step 3 — Run Real-Time Detection
```bash
python scripts/detect_realtime.py
```
- Opens your webcam
- Detects hand signs in real time
- Shows the predicted sign + confidence on screen
- Press **Q** to quit



##  How It Works

```
Webcam Frame
     |
MediaPipe Hand Landmarker
     |
21 Hand Landmarks (x, y coordinates)
     |
Random Forest Classifier
     |
Predicted Sign + Confidence % with charts
```



##  Dependencies

- [OpenCV](https://opencv.org/) — Webcam capture and display
- [MediaPipe](https://mediapipe.dev/) — Hand landmark detection
- [scikit-learn](https://scikit-learn.org/) — Random Forest classifier
- [NumPy](https://numpy.org/) — Data processing

Install all with:
```bash
pip install -r requirements.txt
```



##  Tips to Improve Accuracy

- Collect **200+ images** per sign instead of 50
- Make sure your hand is well-lit and clearly visible
- Try different backgrounds while collecting images
- Normalize landmarks relative to the wrist position for better generalization





