import cv2
import torch
import numpy as np
import torch.nn as nn
import os

# ===== MODEL =====
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*5*5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64*5*5)
        return self.fc(x)

# Load trained model
model = DigitCNN()
model.load_state_dict(torch.load("models/digit_model.pth", map_location="cpu"))
model.eval()

# ===== DIGIT NORMALIZATION =====
def preprocess_digit(img):
    h, w = img.shape
    size = max(h, w) + 20
    canvas = np.zeros((size, size), dtype=np.uint8)

    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = img

    canvas = cv2.resize(canvas, (28,28))

    kernel = np.ones((2,2), np.uint8)
    canvas = cv2.dilate(canvas, kernel, iterations=1)

    canvas = canvas / 255.0
    canvas = (canvas - 0.5) / 0.5
    canvas = canvas.reshape(1,1,28,28)

    return torch.tensor(canvas, dtype=torch.float32)

# ===== OCR PIPELINE =====
def extract_digits(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("Image not found")
        return [], []

    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # remove notebook lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    remove_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.subtract(thresh, remove_lines)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if 20 < w < 200 and 40 < h < 200:
            digit_boxes.append((x,y,w,h))

    digit_boxes = sorted(digit_boxes, key=lambda b: b[0])

    digits = []
    confidences = []

    for (x,y,w,h) in digit_boxes:
        digit_img = thresh[y:y+h, x:x+w]
        digit_tensor = preprocess_digit(digit_img)

        output = model(digit_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

        digits.append(str(pred.item()))
        confidences.append(round(conf.item(),2))

    return digits, confidences

# ===== RUN OCR =====
image_path = "test4.jpeg"

digits, conf = extract_digits(image_path)

result_text = f"Detected number: {''.join(digits)}\nConfidence per digit: {conf}"

print(result_text)

# ===== SAVE OUTPUT =====
os.makedirs("outputs", exist_ok=True)

with open("outputs/result.txt", "w") as f:
    f.write(result_text)

print("Results saved in outputs/result.txt")