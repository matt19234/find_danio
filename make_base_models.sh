#!/bin/bash
mkdir -p base_models
for model in yolov8n-obb yolov8s-obb yolov8m-obb; do
    python -c "from ultralytics import YOLO; YOLO('${model}.pt').save('base_models/${model}.pt')"
done
