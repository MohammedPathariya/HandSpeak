🖐️ HandSpeak: Real-Time Sign Language Detection
HandSpeak is a lightweight, real-time sign language detection system that uses classical machine learning and hand landmarks to classify static gestures, proving that powerful results don't always require deep learning.

🚀 Experience the Live Demo
See the model in action with your own webcam. The frontend sends landmark data to the live backend for classification.

https://handspeak-blush.vercel.app/

💡 My Motivation
Sign language has always fascinated me—it's an expressive, elegant, and deeply human form of communication. The common assumption is that building a system to recognize it in real-time requires complex deep learning models, massive image datasets, and powerful hardware.

That got me thinking...

"What if I could sidestep the complexity? Can I build a functional sign language detector using a more clever, lightweight approach, without ever feeding a single image to the model?"

HandSpeak is my exploration of that idea. Instead of using resource-heavy image processing, this project leverages Google's MediaPipe to extract just the essential data: the 3D coordinates of 21 key points on the hand. By training a simple machine learning model on this clean, structured landmark data, HandSpeak became a fascinating journey into building an end-to-end ML system that is both surprisingly effective and incredibly efficient.

✨ Key Features
Real-Time Hand Tracking: Uses OpenCV and MediaPipe to instantly detect and track hand landmarks from a live webcam feed.

Image-Free Machine Learning: The model is trained exclusively on 63 data points (21 landmarks x 3 coordinates), making it extremely fast and lightweight.

Live Gesture Classification: Classifies six fundamental static American Sign Language (ASL) gestures: Hello, Goodbye, Please, Thank You, Yes, and No.

Interactive Web Interface: A simple and intuitive frontend built with vanilla HTML, CSS, and JavaScript that visualizes the process.

Fully Deployed API: A Flask backend serves the trained Scikit-learn model, ready for integration into any application.

🛠️ Tech Stack & Architecture
Machine Learning: Scikit-learn, Google MediaPipe, OpenCV, Pandas, NumPy

Backend API: Flask, Gunicorn, Joblib

Frontend: HTML, CSS, JavaScript (with Fetch API)

Deployment:

Backend: Hugging Face Spaces for its generous free-tier CPU/RAM.

Frontend: Vercel for high-performance static hosting.

System Architecture Diagram
[ User on Vercel Frontend ]
       |
       | 1. JS captures Webcam & extracts landmarks via MediaPipe
       |
       | 2. Sends only the [63] landmark coordinates (JSON)
       V
[ Hugging Face Space (Backend API) ]
       |
       | 3. Flask app receives coordinates
       |
       | 4. Scikit-learn model predicts the sign
       |
       | 5. Returns the predicted label (e.g., "Hello")
       V
[ User sees the prediction on the Frontend ]

⚙️ How It Works
The project follows a simple yet powerful pipeline:

Video Capture (Client-Side): The frontend uses JavaScript to capture the video feed from the user's webcam.

Landmark Extraction (Client-Side): Each video frame is processed in the browser by the MediaPipe JavaScript library, which identifies any hands and returns the 21 key landmarks (wrist, knuckles, fingertips, etc.) as 3D coordinates.

API Request: This 21x3 landmark array is flattened into a single vector of 63 features. This vector is sent as a JSON payload to the backend API.

Prediction (Server-Side): The Flask backend receives the 63-feature vector and feeds it into the trained Random Forest model.

API Response: The model predicts the sign, and the backend sends the label (e.g., "Thank You") back to the frontend.

Display: The frontend displays the predicted label to the user in real-time.

🔬 Data & Model Training
The model was trained on a custom dataset of hand gestures I captured using the included scripts.

Data Collection
The data_collection.py script uses OpenCV and MediaPipe to capture and save landmark data. Pressing a key saves the current hand landmarks to a file, labeled with the corresponding sign.

Key Press

Sign Label

1

Hello

2

Goodbye

3

Please

4

Thank You

5

Yes

6

No

Model Performance
I evaluated several classical machine learning models. Random Forest provided the best balance of accuracy and speed for this landmark-based dataset.

Model

Accuracy

Notes

Random Forest

~92%

Best overall performance. Chosen model.

XGBoost

~86%

A close second, but slightly slower.

Support Vector (SVM)

~83%

Struggled to distinguish similar hand shapes.

MLP (Neural Network)

⚠️ 100%

Clearly overfit on this small, simple dataset.

The final trained model is saved as sign_model.pkl.

🚀 Getting Started Locally
Prerequisites
Python 3.8+

A webcam

1. Clone the Repository
git clone [https://github.com/MohammedPathariya/HandSpeak.git](https://github.com/MohammedPathariya/HandSpeak.git) # Or your repo URL
cd HandSpeak

2. Install Dependencies
Using a virtual environment is highly recommended.

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required packages
pip install -r requirements.txt

3. Run Real-Time Prediction
To see the model in action on your own machine:

python realtime_predict.py

A window with your webcam feed will open. Show one of the learned signs to the camera, and the model will display the prediction. Press 'q' to quit.

🚧 Limitations & The Road Ahead
Static Signs Only: The current model only recognizes static hand poses and cannot interpret dynamic gestures (e.g., signs that involve motion).

Limited Vocabulary: The model is trained on only six simple, distinct signs.

Sensitivity: Performance can be affected by hand orientation, lighting, and occlusions.

Future improvements could include:

Expanding the Vocabulary: Training the model on the full alphabet and more conversational signs.

Recognizing Dynamic Gestures: Using a recurrent model like an LSTM on sequences of landmarks to understand signs that involve movement.

Two-Handed Signs: Enhancing the feature extraction and model to recognize signs that require both hands.

🙏 Acknowledgements
A huge thank you to the developers of Google MediaPipe for their incredible and easy-to-use hand tracking solution, which is the cornerstone of this project.

The core of the machine learning is powered by the robust and elegant Scikit-learn library.
