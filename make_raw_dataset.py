import cv2
import os
import random
import argparse
import math
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("video")
parser.add_argument("dataset")
parser.add_argument("--num_train", type=int, default=400)
parser.add_argument("--num_val", type=int, default=100)
parser.add_argument("--block_length", type=int, default=1)
args = parser.parse_args()

if args.block_length < 1:
    raise ValueError("block length must be >= 1")

N_TRAIN_BLOCKS = math.ceil(args.num_train / args.block_length)
N_VAL_BLOCKS = math.ceil(args.num_val / args.block_length)
N_BLOCKS = N_TRAIN_BLOCKS + N_VAL_BLOCKS

os.makedirs(args.dataset + "/train", exist_ok=True)
os.makedirs(args.dataset + "/val", exist_ok=True)

cap = cv2.VideoCapture(args.video)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {total_frames}")

if N_BLOCKS * args.block_length > total_frames:
    raise ValueError(
        f"Video has only {total_frames} frames, but "
        f"{N_BLOCKS} non-overlapping sampled blocks of length "
        f"{args.block_length} require at least {N_BLOCKS * args.block_length}"
    )

# Same evenly-spaced sampling logic as before, but now these are block starts.
# For block_length=1 this is identical to the old `indices`.
block_starts = [
    int(i * total_frames / N_BLOCKS)
    for i in range(N_BLOCKS)
]

blocks = []
for start_idx in tqdm(block_starts, desc="Extracting blocks"):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    block = []
    for offset in range(args.block_length):
        ret, frame = cap.read()
        if not ret:
            break

        block.append((start_idx + offset, frame))

    blocks.append(block)

cap.release()

random.shuffle(blocks)

train_blocks = blocks[:N_TRAIN_BLOCKS]
val_blocks = blocks[N_TRAIN_BLOCKS:N_TRAIN_BLOCKS + N_VAL_BLOCKS]

# Flatten, clipping the final block where necessary to get exact counts.
train_frames = [
    frame
    for block in train_blocks
    for frame in block
][:args.num_train]

val_frames = [
    frame
    for block in val_blocks
    for frame in block
][:args.num_val]

num_digits = len(str(total_frames))

for i, frame in tqdm(train_frames, desc="Writing train"):
    cv2.imwrite(args.dataset + f"/train/frame_{i:0{num_digits}d}.jpg", frame)

for i, frame in tqdm(val_frames, desc="Writing val"):
    cv2.imwrite(args.dataset + f"/val/frame_{i:0{num_digits}d}.jpg", frame)

print(f"Train: {len(train_frames)} frames, Val: {len(val_frames)} frames")
