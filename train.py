import argparse
from ultralytics import YOLO
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("base_model")
parser.add_argument("data_path")
parser.add_argument("model_path")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

model = YOLO(args.base_model)
model.train(
    data=args.data_path,
    epochs=500,
    imgsz=1920,
    device="cuda",
    dropout=0.1,
    weight_decay=0.0005,
    patience=0,
)

# model.save(args.model_path)
shutil.copy2(model.trainer.best, args.model_path)
