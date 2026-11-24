from pathlib import Path
from ultralytics import YOLO

output_dir = Path("base_models")
output_dir.mkdir(parents=True, exist_ok=True)

models = ["yolov8n-obb", "yolov8s-obb", "yolov8m-obb"]

for model_name in models:
    # Initialize YOLO (downloads pretrained weights if not cached)
    model = YOLO(f"{model_name}.pt")
    
    # Save a copy to the base_models directory
    model.save(output_dir / f"{model_name}.pt")
