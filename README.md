#  Face Recognition

A Python-based face recognition system using classical computer vision techniques (SVM + PCA) and OpenCV, designed to recognize faces from a standard dataset or webcam in real-time.

---

##  Project Overview

- **Objective**: Recognize and identify faces using image input or live webcam feed.
- **Techniques**:
  - **Principal Component Analysis (PCA)** for dimensionality reduction (eigenfaces).
  - **Support Vector Machine (SVM)** classifier for face recognition.
  - **OpenCV** for image processing and webcam access.
- **Dataset**: Utilizes the AT&T “ORL” face dataset (40 people, 10 images per person).

---

##  Features

-  **Batch Recognition**: Process a folder of test images and output predictions.
-  **Real-Time Webcam Recognition**: Run live face detection and recognition through your webcam.
-  **Eigenface Visualization**: Optionally generate and view eigenfaces to understand PCA components.
-  **Tunable Performance**: Easily adjust training parameters like the number of principal components.

---

##  Tech Stack

- **Language**: Python 3.x  
- **Libraries**:
  - `opencv-python` – Image capture and processing
  - `numpy`, `scipy` – Numerical operations
  - `scikit-learn` – PCA + SVM
  - `joblib` – Model serialization

---

##  Setup & Installation

### Prerequisites
- Python 3.7+
- pip-enabled environment

### Instructions

1. **Clone the repo**:
   ```bash
   git clone https://github.com/sobiamaqbool/Face_recognition.git
   cd Face_recognition

2. **Install dependencies**:
pip install -r requirements.txt
3. **Prepare dataset**:
Download the AT&T dataset (“orl_faces”).
Organize into data/train and data/test directories by person.
4. **Train the model**:
python train.py \
  --data-dir data/train \
  --n-components 50 \
  --model-out models/svm_pca_face.pkl
5. **Run batch recognition**:
python recognize.py \
  --model models/svm_pca_face.pkl \
  --data-dir data/test
6. **Use real-time webcam recognition**:
python recognize_webcam.py \
  --model models/svm_pca_face.pkl
 ## Customization
--n-components: Adjust PCA components (tradeoff between performance & accuracy).
--C, --kernel, --gamma: Tune the SVM classifier parameters.
Supported extensions: .jpg, .png, etc.
##  Results & Metrics
Output includes:
Accuracy and classification report (precision, recall)
Confusion matrix visualization
Live webcam display with bounding boxes and predicted labels
