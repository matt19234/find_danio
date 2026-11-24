from collections import Counter
from ultralytics import YOLO
from pathlib import Path
from copy import deepcopy

output_dir = Path("base_models")

src_model = YOLO(output_dir / "yolov8n-obb.pt")
dst_model = YOLO(output_dir / "yolov8n-p2-obb.yaml", task="obb")

src, dst = src_model.model, dst_model.model

src_sd = src.state_dict()
dst_sd = dst.state_dict()

transfer = {}


def try_transfer(src_key, dst_key):
    if dst_key not in dst_sd:
        print(f"NO TARGET: {src_key} -> {dst_key}")
        return False

    if src_sd[src_key].shape != dst_sd[dst_key].shape:
        print(
            f"SHAPE MISMATCH: {src_key} {tuple(src_sd[src_key].shape)} "
            f"-> {dst_key} {tuple(dst_sd[dst_key].shape)}"
        )
        return False

    transfer[dst_key] = src_sd[src_key]
    return True


# -------------------------------------------------------
# 1. Backbone + unchanged top-down neck: 0..15
# -------------------------------------------------------

for src_key in src_sd:
    if any(src_key.startswith(f"model.{i}.") for i in range(16)):
        try_transfer(src_key, src_key)


# -------------------------------------------------------
# 2. Bottom-up neck shifted by +6
#    old 16..21 -> new 22..27
# -------------------------------------------------------

for old_i in range(16, 22):
    new_i = old_i + 6

    old_prefix = f"model.{old_i}."
    new_prefix = f"model.{new_i}."

    for src_key in src_sd:
        if src_key.startswith(old_prefix):
            dst_key = new_prefix + src_key[len(old_prefix):]
            try_transfer(src_key, dst_key)


# -------------------------------------------------------
# 3. OBB regression head only
#    old P3/P4/P5 -> new P3/P4/P5
# -------------------------------------------------------

for old_scale in range(3):
    new_scale = old_scale + 1

    old_prefix = f"model.22.cv2.{old_scale}."
    new_prefix = f"model.28.cv2.{new_scale}."

    for src_key in src_sd:
        if src_key.startswith(old_prefix):
            dst_key = new_prefix + src_key[len(old_prefix):]
            try_transfer(src_key, dst_key)


# Shared DFL regression layer
old_prefix = "model.22.dfl."

for src_key in src_sd:
    if src_key.startswith(old_prefix):
        dst_key = "model.28.dfl." + src_key[len(old_prefix):]
        try_transfer(src_key, dst_key)


# -------------------------------------------------------
# Actually load the selected tensors
# -------------------------------------------------------

result = dst.load_state_dict(transfer, strict=False)

print(f"\nTransferred {len(transfer)}/{len(dst_sd)} state-dict items")
print(f"Missing target items: {len(result.missing_keys)}")
print(f"Unexpected items: {len(result.unexpected_keys)}")

loaded_keys = set(transfer.keys())
all_keys = set(dst.state_dict().keys())
reported_missing = set(result.missing_keys)

not_transferred = all_keys - loaded_keys

print("state_dict entries:", len(all_keys))
print("transferred:", len(loaded_keys))
print("actually not transferred:", len(not_transferred))
print("reported missing:", len(reported_missing))

unreported = not_transferred - reported_missing

print("\nNot transferred but not reported missing:", len(unreported))
for k in sorted(unreported):
    print(k)

untransferred = set(dst.state_dict()) - set(transfer)

groups = Counter()

for k in untransferred:
    if k.startswith("model.18."):
        groups["new P2 C2f (18)"] += 1
    elif k.startswith("model.19."):
        groups["new P2->P3 Conv (19)"] += 1
    elif k.startswith("model.21."):
        groups["new P3 reconstruction (21)"] += 1
    elif k.startswith("model.28.cv2.0."):
        groups["new P2 bbox head"] += 1
    elif k.startswith("model.28.cv2."):
        groups["other bbox head"] += 1
    elif k.startswith("model.28.cv3."):
        groups["classification head"] += 1
    elif k.startswith("model.28.cv4."):
        groups["angle head"] += 1
    elif "num_batches_tracked" in k:
        groups["BN counters"] += 1
    else:
        groups["OTHER"] += 1

for name, n in groups.items():
    print(f"{name:30s} {n}")

named_params = dict(dst.named_parameters())

total = sum(p.numel() for p in dst.parameters())
transferred = sum(
    named_params[k].numel()
    for k in transfer
    if k in named_params
)

print(f"Transferred params: {transferred:,}/{total:,}")
print(f"Fraction: {100 * transferred / total:.2f}%")

# -------------------------------------------------------
# IMPORTANT: make the custom checkpoint identify itself as OBB
# -------------------------------------------------------

dst_model.task = "obb"
dst.task = "obb"

dst_model.overrides["task"] = "obb"
dst.args["task"] = "obb"

# Sanity checks BEFORE saving.
assert dst_model.task == "obb"
assert dst.task == "obb"
assert dst.args["task"] == "obb"

dst_model.ckpt = {
    "model": deepcopy(dst).half(),
    "ema": None,
    "optimizer": None,
    "train_args": {
        "task": "obb",
        "model": "yolov8n-p2-obb.pt",
    },
}

dst_model.save(output_dir / f"yolov8n-p2-obb.pt")
