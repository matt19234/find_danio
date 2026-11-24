"""Add test frames without changing the train/val split from make_raw_dataset.py.

Pass the same video, dataset directory, and block_length used originally.
The dataset directory must contain the original train/ and val/ image folders,
with every extracted frame retained under its original frame_<index> filename.
"""

import argparse
from bisect import bisect_right
from pathlib import Path
import random
import re


def read_frame_indices(directory):
    if not directory.is_dir():
        raise ValueError(f"Missing split directory: {directory}")
    indices = []
    # The original script writes JPEGs; label files are deliberately ignored.
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        match = re.fullmatch(r"frame_(\d+)", path.stem)
        if not match:
            raise ValueError(f"Unexpected frame filename: {path}")
        indices.append(int(match.group(1)))
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate frame indices in {directory}")
    return sorted(indices)


def recover_blocks(train_indices, val_indices, total_frames, block_length):
    """Recover full [start, end) blocks, including clipped final blocks."""
    if block_length < 1 or total_frames < 1:
        raise ValueError("block_length and total_frames must be positive")
    counts = [len(train_indices), len(val_indices)]
    block_counts = [(n + block_length - 1) // block_length for n in counts]
    n_blocks = sum(block_counts)
    if n_blocks == 0:
        raise ValueError("No existing train/val frames found")
    if n_blocks * block_length > total_frames:
        raise ValueError("Existing frame counts are inconsistent with this video/block_length")

    # Match the original script's start calculation exactly.
    starts = [int(i * total_frames / n_blocks) for i in range(n_blocks)]
    split_blocks = []
    owners = set()
    for name, indices, expected in zip(
        ("train", "val"), (train_indices, val_indices), block_counts
    ):
        grouped = {}
        for index in indices:
            j = bisect_right(starts, index) - 1
            if j < 0 or index >= starts[j] + block_length:
                raise ValueError(
                    f"{name}/frame_{index} does not fit the original block layout. "
                    "Check the video, block_length, and that no frames were removed."
                )
            grouped.setdefault(starts[j], []).append(index)

        if len(grouped) != expected:
            raise ValueError(f"Cannot reconstruct {name} blocks from the existing frames")
        partial = 0
        for start, indices_in_block in grouped.items():
            if sorted(indices_in_block) != list(range(start, start + len(indices_in_block))):
                raise ValueError(f"Missing or duplicate frames in {name} block at {start}")
            partial += len(indices_in_block) < block_length
            if start in owners:
                raise ValueError(f"Train and val share the block at frame {start}")
            owners.add(start)
        if partial > 1:
            raise ValueError(f"More than one partial block in {name}; original frames may be missing")
        split_blocks.append([(s, s + block_length) for s in sorted(grouped)])
    return tuple(split_blocks)


def choose_test_blocks(total_frames, block_length, num_test, excluded, rng):
    """Tile unused gaps with non-overlapping full blocks, then shuffle."""
    candidates = []
    cursor = 0
    for start, end in sorted(excluded) + [(total_frames, total_frames)]:
        candidates.extend(range(cursor, start - block_length + 1, block_length))
        cursor = max(cursor, end)

    required = (num_test + block_length - 1) // block_length
    if required > len(candidates):
        raise ValueError(
            f"Requested {num_test} test frames ({required} blocks), but only "
            f"{len(candidates)} unused full blocks of length {block_length} fit"
        )
    rng.shuffle(candidates)
    return candidates[:required]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--block_length", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.block_length < 1 or args.num_test < 1:
        parser.error("block_length and num_test must be >= 1")

    import cv2
    from tqdm import tqdm

    train_indices = read_frame_indices(args.dataset / "train")
    val_indices = read_frame_indices(args.dataset / "val")
    test_dir = args.dataset / "test"
    if test_dir.exists() and (not test_dir.is_dir() or any(test_dir.iterdir())):
        raise ValueError(f"Test destination must be absent or empty: {test_dir}")

    cap = cv2.VideoCapture(args.video)
    written = []
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {args.video}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        train_blocks, val_blocks = recover_blocks(
            train_indices, val_indices, total_frames, args.block_length
        )
        test_starts = choose_test_blocks(
            total_frames, args.block_length, args.num_test,
            train_blocks + val_blocks, random.Random(args.seed),
        )
        print(f"Total frames: {total_frames}")
        print(f"Existing train blocks: {len(train_blocks)}, val blocks: {len(val_blocks)}")
        print(f"Selected test blocks: {len(test_starts)}")
        test_dir.mkdir(parents=True, exist_ok=True)
        num_digits = len(str(total_frames))
        with tqdm(total=args.num_test, desc="Writing test", unit="frame") as pbar:
            for start in test_starts:
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
                    raise RuntimeError(f"Could not seek to frame {start}")
                # Clip only the final selected block, as in the original script.
                count = min(args.block_length, args.num_test - len(written))
                for offset in range(count):
                    ok, frame = cap.read()
                    index = start + offset
                    if not ok:
                        raise RuntimeError(f"Could not read frame {index}")
                    path = test_dir / f"frame_{index:0{num_digits}d}.jpg"
                    if path.exists():
                        raise FileExistsError(f"Refusing to overwrite {path}")
                    written.append(path)
                    if not cv2.imwrite(str(path), frame):
                        raise RuntimeError(f"Could not write {path}")
                    pbar.update(1)
    except BaseException:
        # Remove this invocation's partial output if extraction fails.
        for path in written:
            path.unlink(missing_ok=True)
        raise
    finally:
        cap.release()

    print(f"Test: {len(written)} frames")


if __name__ == "__main__":
    main()
