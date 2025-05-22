# Face_recognition system using PCA and SVM

This project is a complete face recognition system built with Python, OpenCV, scikit-learn, and Tkinter. It enables:
- Face recognition via webcam using PCA (Principal Component Analysis) and SVM (Support Vector Machine).
- Adding and removing faces to/from the dataset using a GUI.
- Viewing model diagnostics including PCA visualizations and reconstruction.
- Achieving high recognition accuracy through a trained classifier pipeline.

## Features

 - Real-Time Face Recognition using webcam
 - Dimensionality Reduction with PCA
 - Classification using SVM with hyperparameter tuning via GridSearchCV
 - Eigenface Visualization and reconstructed faces
 - Interactive GUI for adding/removing faces
 - Performance Metrics: Accuracy score and confusion matrix

## How It Works

# Load Dataset:
Each person has their own subfolder containing grayscale face images.
# Preprocessing:
Resize each image to 100x100 pixels.
Flatten into a 1D vector.
Normalize with StandardScaler.
# PCA:
Reduce dimensionality to 100 principal components.
Capture ~95% of variance in data.
# SVM Training:
Hyperparameter tuning via GridSearchCV.
Uses an RBF kernel for non-linear decision boundaries.
# Testing:
Accuracy and confusion matrix printed.
Eigenfaces and reconstructed face examples plotted.
# GUI:
Add new face data via webcam.
Remove individuals from dataset.
Launch webcam for real-time recognition.

# Limitations & Suggestions

Ensure well-lit face images and consistent image size.
At least 5–10 images per person are recommended.
Future improvements may include:
Face alignment and cropping automation
Use of face detection before recognition
Support for saving/loading trained models
