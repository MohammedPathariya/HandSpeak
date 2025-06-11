# 🖐️ HandSpeak – Real-Time Sign Language Detection

[](https://handspeak-blush.vercel.app/)
[](https://pathariyamohammed-handspeak-backend.hf.space)
[](https://www.python.org/)
[](https://scikit-learn.org/)

A lightweight, real-time sign language detection system that uses machine learning and hand landmarks, proving that powerful results don't always require deep learning.

-----

### **[► View the Live Demo Here](https://handspeak-blush.vercel.app/)**

-----

## 📌 Overview & Motivation

Sign language has always fascinated me — it's expressive, elegant, and deeply human. But building a system that can recognize sign language in real-time is usually seen as something only possible with deep learning, complex image processing, and huge datasets.

That got me thinking...

> *"Can I build a basic, functional Sign Language Detection system using just simple machine learning — without going deep into CNNs or image-heavy models?"*

That's exactly what this project is about. Instead of using raw images, I leveraged **MediaPipe's** hand landmark detection to work directly with clean, structured data — the 3D coordinates of 21 key points on the hand. This project became a fun exploration of building an end-to-end ML system that is both lightweight and effective.

-----

## ✨ Features

  - **Real-Time Hand Detection**: Instantly detects and isolates hand(s) from a webcam feed using OpenCV.
  - **21-Point Landmark Tracking**: Utilizes Google's MediaPipe to extract 21 3D coordinates for each hand.
  - **Live Gesture Classification**: Classifies static hand gestures for 6 basic signs: *Hello, Goodbye, Please, Thank You, Yes, and No*.
  - **Image-Free ML Model**: The model is trained exclusively on landmark data, making it incredibly fast and lightweight.
  - **Web-Based Interface**: A simple and intuitive frontend built with HTML/CSS/JS that interacts with a deployed backend.
  - **Deployable API**: A Flask backend serves the trained model, ready for integration into other applications.

-----

## 🧰 Technology Stack

  - **Machine Learning**: Scikit-learn, MediaPipe, OpenCV, Pandas, NumPy
  - **Backend API**: Flask, Joblib
  - **Frontend**: HTML, CSS, JavaScript (with Fetch API)
  - **Deployment**:
      - **Backend**: [Hugging Face Spaces](https://pathariyamohammed-handspeak-backend.hf.space)
      - **Frontend**: [Vercel](https://handspeak-blush.vercel.app/)

-----

## ⚙️ How It Works

The project follows a simple yet powerful pipeline:

1.  **Video Capture**: OpenCV captures the video feed from the webcam.
2.  **Hand Landmark Detection**: Each frame is processed by MediaPipe, which identifies any hands and returns the 21 key landmarks (wrist, knuckles, fingertips, etc.) as 3D coordinates (x, y, z).
3.  **Data Preprocessing**: The 21x3 landmark array is flattened into a single vector of 63 features. This vector represents the unique spatial arrangement of the hand for a given sign.
4.  **Prediction**: This 63-feature vector is sent to the trained Random Forest model.
5.  **Classification**: The model predicts the sign and returns the corresponding label (e.g., "Hello").
6.  **Display**: The predicted label is overlaid on the live webcam feed.

-----

## 📂 Project Structure

```
handspeak/
├── captured_data/
│   ├── images/              # (Optional) Saved webcam images for reference
│   └── landmarks/           # Captured landmark data as .npy files
│
├── data_collection.py       # Script to capture and save landmark data
├── realtime_predict.py      # Script to run real-time prediction locally
├── train_model.py           # Script to train the model from landmark data
│
├── app.py                   # Flask backend API
├── sign_model.pkl           # Saved Random Forest model
├── label_classes.npy        # Saved label classes
│
├── landmarks_dataset.csv    # The final prepared dataset for training
└── requirements.txt         # Project dependencies
```

-----

## 🚀 Getting Started Locally

### Prerequisites

  - Python 3.8+
  - A webcam

### 1\. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2\. Install Dependencies

It's recommended to use a virtual environment.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required packages
pip install -r requirements.txt
```

### 3\. Run Real-Time Prediction

To see the model in action with your webcam:

```bash
python realtime_predict.py
```

A window with your webcam feed will open. Show one of the learned signs to the camera, and the model will display the prediction. Press **'q'** to quit.

-----

## 🔬 Data & Model Training

The model was trained on a custom dataset of hand gestures, with each sign corresponding to a numeric label.

### Data Collection & Labeling

You can collect your own data using the `data_collection.py` script.

| Key Press    | Sign Label  |
| :----------- | :---------- |
| **1** | `Hello`     |
| **2** | `Goodbye`   |
| **3** | `Please`    |
| **4** | `Thank You` |
| **5** | `Yes`       |
| **6** | `No`        |

### Model Performance

Several models were evaluated, with Random Forest providing the best balance of accuracy and speed for this dataset.

| Model                 | Accuracy | Notes                                  |
| --------------------- | :------: | -------------------------------------- |
| **Random Forest** | **\~92%** | **Best overall performance and chosen model.** |
| XGBoost               |  \~86%    | A close second, slightly slower.       |
| Support Vector (SVM)  |  \~83%    | Struggled to distinguish similar signs.|
| MLP                   |  ⚠️ 100% | Clearly overfit on the small dataset.  |

The final trained model is saved as `sign_model.pkl`.

-----

## ☁️ API Endpoint

The backend is a Flask API hosted on Hugging Face Spaces that serves predictions.

  - **URL**: `https://pathariyamohammed-handspeak-backend.hf.space/predict`
  - **Method**: `POST`
  - **Body**: A JSON payload containing a 63-element array of the hand landmarks.

**Example Request:**

```json
{
  "landmarks": [0.51, 0.62, -0.01, ..., 0.88, 0.91, -0.05]
}
```

**Success Response (200):**

```json
{
  "prediction": "Thank You"
}
```

-----

## 🚧 Limitations & Future Work

  - **Static Signs Only**: The current model only recognizes static hand poses and cannot interpret dynamic gestures (e.g., signs that involve motion).
  - **Limited Vocabulary**: The model is trained on only six simple signs.
  - **Sensitivity**: Performance can be affected by hand orientation, lighting, and occlusions.

Future improvements could include:

  - **Expanding the Vocabulary**: Adding more signs to the dataset.
  - **Recognizing Dynamic Gestures**: Using recurrent models like LSTMs to understand sequences of landmarks over time.
  - **Two-Handed Signs**: Enhancing the model to recognize signs that require both hands.

-----

## 🤝 Contributing

Contributions are welcome\! If you'd like to help improve HandSpeak, please fork the repository and submit a pull request. You can start by tackling any of the "Future Work" items listed above.

-----

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

-----

## 🙏 Acknowledgements

  - A huge thank you to the developers of **[Google MediaPipe](https://mediapipe.dev/)** for their incredible and easy-to-use hand tracking solution.
  - The core of the machine learning is powered by **[Scikit-learn](https://scikit-learn.org/)**.
