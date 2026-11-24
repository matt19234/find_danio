import cv2
import numpy as np
import argparse
from ultralytics import YOLO
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import re
import yaml
from ultralytics.utils import DATASETS_DIR

# Radius of rendered circles
CIRCLE_RAD = 10

# Max speed of fish (pixels/frame)
MAX_SPEED = 40

# Maximum error of fish center detection model (pixels)
MAX_ERROR = 10

# BGR Colour palette (Okabe and Ito)
COLOURS = [
    (115, 158, 0),   # green
    (0, 159, 230),   # orange
    (66, 228, 240),  # yellow
    (167, 121, 204), # purple
    (233, 180, 86),  # light blue
    (178, 114, 0),   # dark blue
    (0, 94, 213),    # red
]

parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("input_video")
parser.add_argument("output_video")
parser.add_argument("--max_fish", type=int, default=5)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--iou", type=float, default=0.2)
parser.add_argument("--stability", type=int, default=3)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument(
    "--time",
    type=str,
    default=None,
    help="Process time range MM:SS-MM:SS",
)
parser.add_argument(
    "--dataset",
    type=Path,
    default=None,
    help="YOLO YAML: evaluate labeled frames from its test split",
)
args = parser.parse_args()

model = YOLO(args.model_path)

cap = cv2.VideoCapture(args.input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if args.time:
    start_str, end_str = args.time.split("-")

    start_m, start_s = map(int, start_str.split(":"))
    end_m, end_s = map(int, end_str.split(":"))

    start_frame = int((start_m * 60 + start_s) * fps)
    end_frame = min(int((end_m * 60 + end_s) * fps), total_frames)

    if start_frame >= end_frame:
        raise ValueError("--time start must be before end")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    max_frames = end_frame - start_frame
else:
    start_frame = 0
    max_frames = total_frames

if args.max_fish > len(COLOURS):
    raise ValueError(f"Please expand colour palette to support {args.max_fish} fish!")

out = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

def read_label_index(path):
    path = path.expanduser().resolve()
    with path.open() as f:
        config = yaml.safe_load(f)

    root = Path(config.get("path") or path.parent).expanduser()
    if not root.is_absolute() and not root.exists():
        root = Path(DATASETS_DIR) / root
    root = root.resolve()

    splits = config.get("test")
    if isinstance(splits, str):
        splits = [splits]
    if not isinstance(splits, list) or not splits:
        raise ValueError("Dataset YAML must have a nonempty test split")

    label_paths = set()

    for split in splits:
        image_dir = (root / split).resolve()
        parts = list(image_dir.parts)
        if "images" not in parts:
            raise ValueError(f"Expected an 'images' component: {image_dir}")

        # Replace the last 'images' directory component with 'labels'.
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = "labels"
        label_dir = Path(*parts)

        if not label_dir.is_dir():
            raise ValueError(f"Label directory does not exist: {label_dir}")

        label_paths.update(
            path.resolve() for path in label_dir.rglob("frame_*.txt")
        )

    index = {}
    for label_path in sorted(label_paths):
        match = re.fullmatch(r"frame_(\d+)", label_path.stem)
        if not match:
            raise ValueError(f"Unexpected label filename: {label_path}")

        frame_number = int(match.group(1))
        if frame_number in index:
            raise ValueError(
                f"Duplicate frame {frame_number}: "
                f"{index[frame_number]} and {label_path}"
            )
        index[frame_number] = label_path

    if not index:
        raise ValueError("No frame_<number>.txt labels found")

    return index

def read_label(label_path, width, height):
    boxes = []
    for line_number, line in enumerate(
        label_path.read_text().splitlines(), start=1
    ):
        if not line.strip():
            continue

        values = np.asarray(line.split(), dtype=float)
        if values.size != 9 or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid OBB: {label_path}:{line_number}")

        # This evaluator assumes your single-class fish dataset.
        if values[0] != 0:
            raise ValueError(f"Unexpected class: {label_path}:{line_number}")

        corners = values[1:].reshape(4, 2)
        if np.any((corners < 0) | (corners > 1)):
            raise ValueError(
                f"Coordinates must be normalized: {label_path}:{line_number}"
            )

        boxes.append(corners * [width, height])

    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4, 2)

def eval_frame(fish_state, curr_t, boxes):
    fresh = fish_state["stable_time"] == curr_t
    points = fish_state["stable_loc"][fresh]

    n_predictions, n_labels = len(points), len(boxes)
    if n_predictions == 0 or n_labels == 0:
        return 0, n_predictions, n_labels

    eligible = np.zeros((n_predictions, n_labels), dtype=bool)
    for i, point in enumerate(points):
        for j, box in enumerate(boxes):
            eligible[i, j] = cv2.pointPolygonTest(
                np.asarray(box, dtype=np.float32),
                (float(point[0]), float(point[1])),
                False,
            ) >= 0

    # Cost -1 for an eligible match, 0 otherwise:
    # minimizing total cost maximizes the number of eligible matches.
    rows, cols = linear_sum_assignment(-eligible.astype(float))
    tp = int(eligible[rows, cols].sum())

    # Ineligible assigned pairs are ignored.
    return tp, n_predictions - tp, n_labels - tp

def get_obs(result):
    if len(result.obb) > 0:
        obb = result.obb
        confs = obb.conf.cpu().numpy()
        sorted_indices = np.argsort(confs)[::-1]

        return np.array([
            obb.xyxyxyxy[idx].cpu().numpy().reshape(4, 2)
            for idx in sorted_indices
        ])

    return np.empty((0, 4, 2))

def get_loc(obs):
    return np.mean(obs, axis = -2)

def init_fish_state(obs):
    if len(obs) < args.max_fish:
        return None

    top_obs = obs[:args.max_fish]
    top_loc = get_loc(top_obs)
    return {
        "loc": np.array(top_loc),                    # likely current location
        "time": np.zeros(len(top_loc)),              # time when loc observed
        "vel": np.zeros_like(top_loc),               # velocity estimated from loc/time
        "stable_loc": np.array(top_loc),             # higher confidence current location
        "stable_time": np.zeros(len(top_loc)),       # time when stable_loc observed
        "fail_time": -np.inf * np.ones(len(top_loc)) # last time when not observed
    }

def update_fish_state(fish_state, obs, curr_t):
    loc = get_loc(obs)
    n = len(fish_state["loc"])
    dt = curr_t - fish_state["time"]
    stable_dt = curr_t - fish_state["stable_time"]

    predicted_loc = fish_state["loc"] + fish_state["vel"] * dt[:, None]
    displacement = np.linalg.norm(fish_state["stable_loc"][:, None, :] - loc[None, :, :], axis = 2)
    max_displacement = MAX_SPEED * stable_dt + MAX_ERROR
    cost = np.linalg.norm(predicted_loc[:, None, :] - loc[None, :, :], axis = 2)
    max_cost = np.max(cost) if cost.size > 0 else 0.0
    cost[displacement > max_displacement[:, None]] = np.inf

    # allow each fish to be assigned no observation
    cost = np.hstack((cost, (max_cost + 1) * np.ones((n, n))))

    # Hungarian algorithm for optimal assignment
    I, J = linear_sum_assignment(cost)

    new_fish_state = {
        "loc": np.copy(fish_state["loc"]),
        "time": np.copy(fish_state["time"]),
        "vel": np.copy(fish_state["vel"]),
        "stable_loc": np.copy(fish_state["stable_loc"]),
        "stable_time": np.copy(fish_state["stable_time"]),
        "fail_time": np.copy(fish_state["fail_time"])
    }
    for i, j in zip(I, J):
        if j < len(loc):
            new_fish_state["loc"][i] = loc[j]
            new_fish_state["time"][i] = curr_t

            if curr_t >= fish_state["fail_time"][i] + args.stability:
                new_fish_state["stable_loc"][i] = new_fish_state["loc"][i]
                new_fish_state["stable_time"][i] = curr_t

            # only estimate velocity if we have a stable estimate of location
            if new_fish_state["stable_time"][i] > fish_state["fail_time"][i]:
                vel = (new_fish_state["loc"][i] - fish_state["loc"][i]) / dt[i]
                speed = np.linalg.norm(vel)
                vel = vel * min(1, MAX_SPEED / speed)
                # geometric avg velocity estimate to smooth model instability
                new_fish_state["vel"][i] = 1/2 * vel + 1/2 * fish_state["vel"][i]
            else:
                new_fish_state["vel"][i] = np.zeros(2)
        else:
            new_fish_state["fail_time"][i] = curr_t
            if curr_t >= fish_state["fail_time"][i] + args.stability:
                new_fish_state["loc"][i] = fish_state["stable_loc"][i]
                new_fish_state["time"][i] = fish_state["stable_time"][i]
                new_fish_state["vel"][i] = np.zeros(2)
    return new_fish_state

def render_frame(frame, fish_state, obs, curr_t):
    for i, loc in enumerate(fish_state["stable_loc"]):
        center = tuple(loc.astype(int))
        cv2.circle(frame, center, CIRCLE_RAD, COLOURS[i], -1)
    return frame

def render_frame_debug(frame, fish_state, obs, curr_t):
    # Draw every raw model OBB:
    # thick black outline underneath, thinner white outline on top
    for box in obs:
        # center = tuple(get_loc(box).astype(int))
        # cv2.circle(frame, center, CIRCLE_RAD // 2, (255, 255, 255), -1)
        # cv2.circle(frame, center, CIRCLE_RAD // 2, (0, 0, 0), 2)

        pts = np.rint(box).astype(np.int32).reshape((-1, 1, 2))

        cv2.polylines(
            frame,
            [pts],
            isClosed=True,
            color=(0, 0, 0),
            thickness=5,
            lineType=cv2.LINE_AA,
        )

        cv2.polylines(
            frame,
            [pts],
            isClosed=True,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    for i, (loc, vel, stable_loc, stable_time) in enumerate(zip(
        fish_state["loc"],
        fish_state["vel"],
        fish_state["stable_loc"],
        fish_state["stable_time"]
    )):
        stable_loc_pt = tuple(np.rint(stable_loc).astype(int))
        opacity = np.clip(
            np.exp(-(curr_t - stable_time) / 10),
            0.0,
            1.0,
        )
        overlay = frame.copy()
        cv2.circle(
            overlay,
            stable_loc_pt,
            CIRCLE_RAD,
            COLOURS[i],
            thickness=-1,
        )
        cv2.addWeighted(
            overlay,
            opacity,
            frame,
            1.0 - opacity,
            0.0,
            dst=frame,
        )

        loc_pt = tuple(np.rint(loc).astype(int))
        arrow = loc + vel * 10
        arrow_pt = tuple(np.rint(arrow).astype(int))
        cv2.arrowedLine(
            frame,
            loc_pt,
            arrow_pt,
            COLOURS[i],
            thickness=2,
            tipLength=0.3,
        )

    return frame

def scores_from_counts(counts):
    tp, fp, fn = np.asarray(counts, dtype=float)
    numerators = np.array([tp, tp, 2 * tp])
    denominators = np.array([tp + fp, tp + fn, 2 * tp + fp + fn])

    return np.divide(
        numerators,
        denominators,
        out=np.full(3, np.nan),
        where=denominators > 0,
    )

label_index = read_label_index(args.dataset) if args.dataset else {}
eval = []
render = render_frame_debug if args.debug else render_frame
fish_state = None
batch_frames = []
with tqdm(total=max_frames, desc="Processing", unit="frame") as pbar:
    frames_processed = 0
    curr_t = 0 
    while frames_processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames_processed += 1
        batch_frames.append(frame)

        if len(batch_frames) == args.batch or frames_processed == max_frames - 1:
            results = model(batch_frames, imgsz=1920, iou=args.iou, verbose=False, device=args.device)
            for frame, result in zip(batch_frames, results):
                obs = get_obs(result)
                if fish_state:
                    fish_state = update_fish_state(fish_state, obs, curr_t)
                else:
                    fish_state = init_fish_state(obs)

                if fish_state:
                    frame_number = start_frame + curr_t
                    label_path = label_index.get(frame_number)
                    if label_path:
                        label = read_label(label_path, w, h)
                        eval.append(eval_frame(fish_state, curr_t, label))

                    out.write(render(frame, fish_state, obs, curr_t))

                curr_t += 1
            pbar.update(len(batch_frames))
            batch_frames = []

cap.release()
out.release()
print("Done")

if eval:
    counts = np.asarray(eval, dtype=np.int64)
    totals = counts.sum(axis=0)
    scores = scores_from_counts(totals)
    names = ("Precision", "Recall", "F1")

    print(f"\nEvaluated labeled frames: {len(counts)}")
    print(f"TP={totals[0]}, FP={totals[1]}, FN={totals[2]}")

    for name, value in zip(names, scores):
        text = f"{value:.4f}" if np.isfinite(value) else "undefined"
        print(f"{name}: {text}")

    if len(counts) < 2:
        print("Skipping CI: fewer than two labeled frames.")
        exit()

    rng = np.random.default_rng(0)
    bootstrap_scores = np.empty((2000, 3))

    for i in range(len(bootstrap_scores)):
        selected = rng.integers(0, len(counts), size=len(counts))
        bootstrap_scores[i] = scores_from_counts(
            counts[selected].sum(axis=0)
        )

    for j, name in enumerate(names):
        samples = bootstrap_scores[:, j]

        if not np.all(np.isfinite(samples)):
            print(f"{name} CI unavailable: undefined bootstrap ratios.")
        elif np.ptp(samples) == 0:
            print(f"{name} CI unavailable: bootstrap distribution is degenerate.")
        else:
            lo, hi = np.quantile(samples, [0.025, 0.975])
            print(f"{name} approximate 95% bootstrap CI: [{lo:.4f}, {hi:.4f}]")
