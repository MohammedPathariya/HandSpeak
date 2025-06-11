# Sign Language Detection using Hand Landmarks

---

## Overview & Motivation

Sign language has always fascinated me — it's expressive, elegant, and deeply human. But building a system that can recognize sign language in real-time is usually seen as something only possible with deep learning, complex image processing, and huge datasets.

That got me thinking...

> *"Can I build a basic, functional Sign Language Detection system using just simple machine learning — without going deep into CNNs or image-heavy models?"*

That's exactly what this project is about.

Instead of using raw images (which require a lot of data and heavy models), I leveraged MediaPipe's hand landmark detection to directly work with clean, structured data — the 3D coordinates of 21 key points on the hand.

This project became a fun exploration of:
- End-to-end ML workflow
- Real-time prediction using webcam
- Lightweight solutions with powerful results
- And proving to myself that sometimes — simple works beautifully.

---

## What Does This Project Do?

- Detects hand(s) from webcam feed
- Tracks 21 hand landmarks using MediaPipe
- Classifies static hand gestures into 6 basic sign language words:
  - Hello
  - Goodbye
  - Please
  - Thank You
  - Yes
  - No
- Displays the predicted sign live on the webcam feed
- Built using a Random Forest ML model trained only on landmark data (no raw images!)

---

## Tech Stack & Tools Used

- Python
- OpenCV → Webcam access
- MediaPipe → Hand Landmark Detection
- Scikit-learn → ML Models
- Pandas, NumPy → Data handling
- Joblib → Saving & Loading models

---

## Signs & Labels Used

| Key Press | Sign Label |
|-----------|------------|
| 1 | Hello |
| 2 | Goodbye |
| 3 | Please |
| 4 | Thank You |
| 5 | Yes |
| 6 | No |

---

## Project Phases

### 1. Data Collection
- Captured images + hand landmark data using webcam.
- MediaPipe tracked 21 landmarks per hand.
- On pressing keys `1` to `6`, the system saved:
  - Image of hand
  - Landmark `.npy` file of coordinates
- Stored data in separate folders for each sign.

```
captured_data/
├── images/         → Saved webcam images (reference only)
├── landmarks/      → Landmark .npy files (used for ML)
```

---

### 2. Data Preparation
- Loaded all `.npy` files.
- Flattened 21x3 landmarks into a 63-length feature vector.
- Labeled data based on folder names.
- Final dataset → `landmarks_dataset.csv`

Example Row:
| x0 | y0 | z0 | ... | x20 | y20 | z20 | label |
|----|----|----|-----|-----|-----|-----|-------|
|... |... |... |...  |...  |...  |...  | hello |

---

### 3. Model Training & Evaluation
Tried multiple ML models to see what works best:

| Model | Accuracy | Remarks |
|-------|-----------|---------|
| Random Forest | ~91.6% | Most balanced & reliable |
| SVM | ~83.3% | Struggled with some signs |
| XGBoost | ~86.1% | Decent, but not better than RF |
| MLP (Neural Net) | 100% | Classic overfitting (tiny dataset!) |

> Final Model Selected → Random Forest Classifier  
Saved as: `sign_model.pkl`  
Label Mapping saved as: `label_classes.npy`

---

### 4. Real-Time Sign Prediction
- Real-time webcam detection using OpenCV + MediaPipe.
- Landmark extraction from live video.
- Predict sign using Random Forest.
- Display prediction on the webcam window.

---

## Running Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run real-time prediction:
```bash
python realtime_predict.py
```

3. Press `q` to quit the webcam window.

---

## Project Folder Structure

```
sign_language_project/
│
├── captured_data/              → Images & landmark data
├── landmarks_dataset.csv       → Final dataset
│
├── data_prep.ipynb             → Data cleaning & preparation
├── model_training.ipynb        → Model building & evaluation
├── realtime_predict.py         → Real-time sign prediction
│
├── sign_model.pkl              → Final trained model
├── label_classes.npy           → Label mapping
│
└── requirements.txt            → Python dependencies
```

---

## Challenges & Learnings Along The Way

- Data collection was repetitive and time-consuming (manual sign capturing).
- Realized quickly that the model worked well only in controlled conditions (same hand, same environment).
- Landmark detection failed occasionally if hand wasn't fully in frame.
- MLP model showing 100% accuracy was a "red flag" moment — overfitting is very real.
- Maintaining clean label mapping across different scripts was trickier than expected.

---

## Why Is It Working So Well?

- Very clean and structured input (landmark data only).
- Single user (me) for all data.
- Consistent lighting, environment, and hand position.
- Very small and well-defined problem (only 6 signs).

---

## But Why Should I Still Be Careful?

- Might fail on other users.
- Different lighting or backgrounds might reduce accuracy.
- Different hand orientations could confuse the model.
- Real-world performance needs more data variety.

---

## Future Plans & Ideas

- Add data from multiple users for better generalization.
- Capture data in varied environments.
- Add more signs & gestures.
- Deploy the system as a simple web app using Flask/Streamlit.
- Explore sequence-based detection for dynamic signs.
- Try deep learning models (LSTM / CNN) once dataset grows.

---

## Final Thoughts

This project started as a small experiment — "Can simple machine learning detect sign language in real-time?"  
Turns out — Yes, it absolutely can (within its limitations).

This was an incredibly fun learning experience covering:
- Real-time ML systems
- Clean end-to-end pipelines
- Small data project challenges
- Working smart → Not just working hard

---

## Author
Made with patience, curiosity, and a lot of key-pressing by  
**Mohammed Johar Pathariya**