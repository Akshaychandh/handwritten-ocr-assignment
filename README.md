# Handwritten Digit OCR using Deep Learning

## Overview

This project implements an Optical Character Recognition (OCR) system for extracting handwritten digits from images.
The solution uses a **Convolutional Neural Network (CNN)** trained on the MNIST dataset combined with image preprocessing and contour-based digit segmentation.

The goal is to demonstrate a practical deep learning–based OCR pipeline capable of recognizing handwritten number sequences from real-world images.

---

## Approach

The system follows a two-stage pipeline:

### 1. Image Processing & Segmentation

* Convert image to grayscale
* Apply adaptive thresholding
* Remove noise and background artifacts
* Detect contours corresponding to digits
* Sort digits from left to right
* Crop individual digit images

### 2. Digit Recognition using Deep Learning

* A CNN trained on the MNIST dataset is used as the classifier
* Each segmented digit is normalized to MNIST format (28×28 pixels)
* The network predicts the digit and provides confidence scores

---

## Model Architecture

The CNN consists of:

* Convolution Layer → ReLU → MaxPooling
* Convolution Layer → ReLU → MaxPooling
* Fully Connected Layer → ReLU
* Output Layer (10 classes)

The model is trained using cross-entropy loss and the Adam optimizer.

---

## Project Structure

```
handwritten_ocr_assignment/
│
├── src/
│   ├── train_model.py        # CNN training script
│   └── predict_digits.py     # OCR prediction pipeline
│
├── models/
│   └── digit_model.pth       # Trained CNN model
│
├── test.jpeg                 # Sample handwritten test image
├── README.md
├── report.docx
```

---

## How to Run

### 1. Install dependencies

```
pip install torch torchvision opencv-python numpy matplotlib
```

### 2. Train the model

```
python src/train_model.py
```

This downloads MNIST and trains the CNN.

### 3. Run OCR on an image

Place your test image in the project root and run:

```
python src/predict_digits.py
```

The script outputs:

* Detected number sequence
* Confidence score per digit

---

## Example Output

```
Detected number: 471936
Confidence per digit: [0.92, 0.96, 0.99, 0.69, 0.95, 0.72]
```

---

## Challenges Faced

* Handwriting styles differ significantly from MNIST samples
* Thin strokes and uneven spacing reduce classification accuracy
* Lighting conditions affect thresholding performance

---

## Possible Improvements

* Fine-tuning the model on additional handwritten samples
* Using data augmentation to simulate real handwriting variations
* Implementing a digit detection neural network instead of contour filtering
* Deploying a transformer-based OCR model for end-to-end recognition

---

## Conclusion

This project demonstrates a functional deep learning–based OCR pipeline that combines classical image processing with CNN classification.
While accuracy varies depending on handwriting style and image quality, the system successfully illustrates the principles of handwritten digit recognition in real-world conditions.
