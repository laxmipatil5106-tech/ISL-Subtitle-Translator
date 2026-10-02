# 🤟 ISL Subtitle Translator

> 🚧 **Project Status: Currently Under Development**

An AI/ML-based project for developing a real-time **Indian Sign Language (ISL) recognition and subtitle system**.

The project aims to assist communication between people who use Indian Sign Language and people who do not understand sign language by recognizing signs through a camera and converting them into real-time text subtitles.

---

## 🎯 Project Objective

The objective of this project is to develop a computer-vision and deep-learning based system that can:

- Detect human pose, face and hand landmarks from a live camera feed.
- Extract meaningful sequential features from ISL signs.
- Recognize trained ISL signs using a deep-learning model.
- Display recognized signs as real-time subtitles.
- Eventually translate recognized signs into multiple languages.
- Eventually provide voice output for recognized or translated text.

The system is being developed incrementally, with model accuracy, vocabulary and real-time performance being improved throughout development.

---

## 🧠 How It Works

The current system follows this pipeline:

```text
Camera Input
     ↓
MediaPipe Holistic
     ↓
Landmark & Feature Extraction
     ↓
30-Frame Sequence
     ↓
Bidirectional LSTM Model
     ↓
ISL Sign Prediction
     ↓
Real-Time Subtitle
Unlike a simple static gesture classifier, the current approach uses a sequence of consecutive frames to capture movement over time.

⚙️ Current Implementation
The current version includes:
Real-time camera input using OpenCV
MediaPipe Holistic for landmark detection
Pose, face and hand landmark extraction
30-frame sequential input
1,662 features per frame
Bidirectional LSTM-based classification
40 trained ISL sign classes
Confidence-based prediction
Real-time sign prediction
Initial real-time subtitle functionality
The recognition model is currently being tested and improved for better accuracy and generalization.

🧩 Feature Extraction
Each frame produces 1,662 features from MediaPipe Holistic landmarks.
Component	Features
Pose landmarks	132
Face landmarks	1,404
Left hand landmarks	63
Right hand landmarks	63
Total	1,662
The system collects 30 consecutive frames to represent the temporal movement of a sign before making a prediction.

🤖 Machine Learning Model
The current recognition model uses a Bidirectional LSTM architecture to process sequential landmark data.
Current Architecture
Input Sequence
     ↓
Bidirectional LSTM (128)
     ↓
Dropout
     ↓
Bidirectional LSTM (64)
     ↓
Dropout
     ↓
Dense Layer (64, ReLU)
     ↓
Dropout
     ↓
Softmax Output
The current model has been trained on 40 sign classes.

🛠️ Technologies Used
Python
OpenCV
MediaPipe
TensorFlow / Keras
NumPy
Scikit-learn
Bidirectional LSTM
Computer Vision
Deep Learning
Git & GitHub

📁 Project Structure
ISL-Subtitle-Translator/
│
├── preprocess_v2.py
├── train_model_v2.py
├── realtime_subtitles_v2.py
├── README.md
└── .gitignore
The trained model and dataset are currently maintained separately during development and are not included in this repository.

🚀 Running the Project
1. Clone the repository
git clone https://github.com/laxmipatil5106-tech/ISL-Subtitle-Translator.git
cd ISL-Subtitle-Translator
2. Create / activate the Python environment
The project is currently being developed and tested using Python 3.12.
python3.12 -m venv signlang312
source signlang312/bin/activate
3. Install dependencies
pip install opencv-python mediapipe tensorflow numpy scikit-learn
4. Run real-time recognition
python3 realtime_subtitles_v2.py
Note: The project is currently under development. The trained model and dataset are not included in this repository at this stage, so additional project files may be required to reproduce the complete current development environment.

🚧 Development Status
This project is actively under development and should not be considered a completed system.
Current development areas
Improving sign recognition accuracy
Expanding the ISL vocabulary
Improving real-time subtitle reliability
Improving model generalization
Testing signs under different conditions
Improving prediction stability
Improving the user interface
Refining the overall recognition pipeline
The current implementation demonstrates the core recognition pipeline, while additional features and improvements are being developed.

🔮 Planned Improvements
Future development is planned to explore:
Larger and more diverse ISL datasets
Recognition of a broader vocabulary
Improved temporal sequence modelling
Sentence-level sign recognition
Improved English translation
Support for additional languages such as Hindi and Marathi
Text-to-speech functionality
Improved real-time performance
Better handling of different lighting and backgrounds
Better handling of different signing speeds
User-friendly application deployment
These features are planned or under exploration and are not currently claimed as completed functionality.

🎓 Project Purpose
This project is being developed as an exploration of:
Artificial Intelligence
Machine Learning
Computer Vision
Deep Learning
Human-Computer Interaction
Accessibility-focused technology
The long-term goal is to explore how AI and computer vision can assist communication between ISL users and people who do not understand sign language.

👩‍💻 Developer
Laxmi Patil
B.Tech Engineering Student
Interests:
AI/ML • Computer Vision • Python • Deep Learning