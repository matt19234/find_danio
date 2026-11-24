import json
import cv2
import numpy as np
import argparse
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("src_dataset")
parser.add_argument("dst_dataset")
args = parser.parse_args()

SRC_DATASET = Path(args.src_dataset)
DST_DATASET = Path(args.dst_dataset)

for split in ["train", "val", "test"]:
    src_dir = SRC_DATASET / split
    if not src_dir.is_dir():
        print(f"Skipping missing split: {src_dir}")
        continue

    dst_images_dir = DST_DATASET / "images" / split
    dst_labels_dir = DST_DATASET / "labels" / split
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    for json_path in src_dir.glob("*.json"):
        with open(json_path) as f:
            data = json.load(f)

        h, w = data["imageHeight"], data["imageWidth"]
        txt_path = dst_labels_dir / json_path.with_suffix(".txt").name

        with open(txt_path, "w") as out:
            for shape in data["shapes"]:
                pts = np.array(shape["points"], dtype=np.float32)

                if len(pts) < 3:
                    print(f"Skipping shape with {len(pts)} points in {json_path.name}")
                    continue

                rect = cv2.minAreaRect(pts)
                corners = cv2.boxPoints(rect)

                corners[:, 0] /= w
                corners[:, 1] /= h

                coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in corners)
                out.write(f"0 {coords}\n")

        # TODO: copy src_dir / <image corresponding to json_path> to
        # dst_images_dir / <image corresponding to json path>
        img_path = json_path.with_suffix(".jpg")
        shutil.copy(img_path, dst_images_dir / img_path.name)

        print(f"Converted {json_path.name} -> {txt_path}")

print("Done")
