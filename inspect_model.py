"""
This script displays stride properties of a trained model
onto a given image to help with troubleshooting model
performance.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("input_path")
parser.add_argument("output_path")
args = parser.parse_args()

yolo_model = YOLO(args.model_path)
model = yolo_model.model

# Inspect actual layer strides from the PyTorch model
if hasattr(model, 'stride') and model.stride is not None:
    strides = model.stride.tolist()
    print(f"Detected Model Strides: {strides}")
else:
    # Fallback to standard YOLO strides [P3, P4, P5]
    strides = [8.0, 16.0, 32.0]
    print(f"Using standard YOLO Strides: {strides}")

# Load image
img = cv2.imread(args.input_path)
if img is None:
    raise FileNotFoundError(f"Could not load image at {args.input_path}")

h, w, _ = img.shape
center_x, center_y = w // 2, h // 2

# Define head specifications (Stride, Color BGR, Label)
head_configs = [
    {"stride": 4,  "color": (255, 0, 255), "name": "P2 (Stride 4)"},   # Magenta (Optional Small Head)
    {"stride": 8,  "color": (0, 255, 0),   "name": "P3 (Stride 8)"},   # Green (Standard Smallest Head)
    {"stride": 16, "color": (255, 255, 0), "name": "P4 (Stride 16)"},  # Cyan
    {"stride": 32, "color": (0, 0, 255),   "name": "P5 (Stride 32)"},  # Red
]

# Draw grid cell boxes near the image center for comparison
offset_x = 0
spacing = 30  # Space out the visual boxes horizontally

for head in head_configs:
    s = head["stride"]
    color = head["color"]
    name = head["name"]

    # Calculate top-left and bottom-right corner for an (s x s) box
    x1 = center_x + offset_x
    y1 = center_y
    x2 = x1 + s
    y2 = y1 + s

    # Draw the grid cell box (2px thick for visibility)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Add label text above each box
    text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    cv2.putText(
        img, 
        name, 
        (x1, y1 - 8), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.4, 
        color, 
        1, 
        cv2.LINE_AA
    )

    offset_x += s + spacing

# Save result
cv2.imwrite(args.output_path, img)
print(f"Saved stride visualization to: {args.output_path}")
