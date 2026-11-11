"""
Badminton AI Coach — Week 1-2: Pose Estimation + Player Tracking
================================================================
Uses YOLOv8-Pose only (no MediaPipe dependency).

Dependencies:
    pip install opencv-python ultralytics numpy

Usage:
    python badminton_tracker.py --input match.mp4 --output output.mp4
    python badminton_tracker.py --input match.mp4 --display
    python badminton_tracker.py --input match.mp4 --output out.mp4 --json stats.json --max-frames 300
"""

import cv2
import numpy as np
import argparse
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Optional
from ultralytics import YOLO


# ─── YOLO-Pose keypoint indices (COCO 17-point skeleton) ──────────────────────
KP = {
    "nose": 0,
    "left_eye": 1,   "right_eye": 2,
    "left_ear": 3,   "right_ear": 4,
    "left_shoulder": 5,  "right_shoulder": 6,
    "left_elbow": 7,     "right_elbow": 8,
    "left_wrist": 9,     "right_wrist": 10,
    "left_hip": 11,      "right_hip": 12,
    "left_knee": 13,     "right_knee": 14,
    "left_ankle": 15,    "right_ankle": 16,
}

SKELETON = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class JointAngles:
    right_elbow:    Optional[float] = None
    left_elbow:     Optional[float] = None
    right_knee:     Optional[float] = None
    left_knee:      Optional[float] = None
    right_shoulder: Optional[float] = None
    left_shoulder:  Optional[float] = None
    right_hip:      Optional[float] = None
    left_hip:       Optional[float] = None

@dataclass
class PlayerFrame:
    track_id:  int
    bbox:      tuple
    center:    tuple
    keypoints: list
    angles:    JointAngles
    frame_idx: int

@dataclass
class FrameData:
    frame_idx: int
    players:   list = field(default_factory=list)


# ─── Geometry ─────────────────────────────────────────────────────────────────

def angle_between(a, b, c) -> float:
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag = math.hypot(*ba) * math.hypot(*bc)
    if mag < 1e-6:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


def extract_angles(kps: list, conf_thresh: float = 0.3) -> JointAngles:
    def pt(name):
        k = kps[KP[name]]
        return (k[0], k[1]), k[2]

    def safe_angle(a_name, b_name, c_name):
        (a, ca), (b, cb), (c, cc) = pt(a_name), pt(b_name), pt(c_name)
        if ca < conf_thresh or cb < conf_thresh or cc < conf_thresh:
            return None
        return angle_between(a, b, c)

    return JointAngles(
        right_elbow    = safe_angle("right_shoulder", "right_elbow",    "right_wrist"),
        left_elbow     = safe_angle("left_shoulder",  "left_elbow",     "left_wrist"),
        right_knee     = safe_angle("right_hip",      "right_knee",     "right_ankle"),
        left_knee      = safe_angle("left_hip",       "left_knee",      "left_ankle"),
        right_shoulder = safe_angle("right_elbow",    "right_shoulder", "right_hip"),
        left_shoulder  = safe_angle("left_elbow",     "left_shoulder",  "left_hip"),
        right_hip      = safe_angle("right_shoulder", "right_hip",      "right_knee"),
        left_hip       = safe_angle("left_shoulder",  "left_hip",       "left_knee"),
    )


# ─── Visualisation ────────────────────────────────────────────────────────────

COLOURS = [
    (0, 210, 255),
    (255, 100,  40),
    (80,  255, 120),
    (255,  60, 200),
]

def colour_for(track_id: int) -> tuple:
    return COLOURS[int(track_id) % len(COLOURS)]


def draw_player(frame, player: PlayerFrame):
    col = colour_for(player.track_id)
    x1, y1, x2, y2 = player.bbox

    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
    cv2.putText(frame, f"P{player.track_id}",
                (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

    kps = player.keypoints
    for a_idx, b_idx in SKELETON:
        a, b = kps[a_idx], kps[b_idx]
        if a[2] < 0.3 or b[2] < 0.3:
            continue
        cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col, 2)

    for kp in kps:
        if kp[2] < 0.3:
            continue
        cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, col, -1)

    ang = player.angles
    lines = []
    if ang.right_elbow    is not None: lines.append(f"R.Elbow  {ang.right_elbow:.0f}deg")
    if ang.left_elbow     is not None: lines.append(f"L.Elbow  {ang.left_elbow:.0f}deg")
    if ang.right_knee     is not None: lines.append(f"R.Knee   {ang.right_knee:.0f}deg")
    if ang.left_knee      is not None: lines.append(f"L.Knee   {ang.left_knee:.0f}deg")
    for i, ln in enumerate(lines):
        cv2.putText(frame, ln, (x1, y2 + 15 + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)


def draw_hud(frame, frame_idx: int, n_players: int):
    cv2.putText(frame, f"Frame {frame_idx:05d}  |  Players: {n_players}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)


# ─── Main pipeline ────────────────────────────────────────────────────────────

def process_video(input_path, output_path, display, conf=0.4, max_frames=None):
    print("[1/2] Loading YOLOv8-Pose model ...")
    model = YOLO("yolov8n-pose.pt")

    print("[2/2] Opening video ...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {input_path}")

    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"    {W}x{H}  {fps:.1f} fps  {total} frames")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    all_frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break

        fd = FrameData(frame_idx=frame_idx)

        results = model.track(frame, persist=True, conf=conf, verbose=False)

        if results and results[0].keypoints is not None:
            boxes   = results[0].boxes
            kp_data = results[0].keypoints.data

            for i, box in enumerate(boxes):
                if box.id is None:
                    continue
                track_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                kps = kp_data[i].cpu().numpy().tolist()
                angles = extract_angles(kps)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                pf = PlayerFrame(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                    keypoints=kps,
                    angles=angles,
                    frame_idx=frame_idx,
                )
                fd.players.append(pf)
                draw_player(frame, pf)

        draw_hud(frame, frame_idx, len(fd.players))
        all_frames.append(fd)

        if writer:
            writer.write(frame)
        if display:
            cv2.imshow("Badminton Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  ... {frame_idx}/{total} frames")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    unique_ids = {p.track_id for fd in all_frames for p in fd.players}
    print(f"\nDone - {frame_idx} frames, {len(unique_ids)} unique player IDs.")
    return all_frames


# ─── Analytics summary ────────────────────────────────────────────────────────

def summarise(all_frames):
    stats = {}
    for fd in all_frames:
        for p in fd.players:
            tid = p.track_id
            if tid not in stats:
                stats[tid] = {j: [] for j in JointAngles.__dataclass_fields__}
            for joint, val in asdict(p.angles).items():
                if val is not None:
                    stats[tid][joint].append(val)

    summary = {}
    for tid, joints in stats.items():
        summary[f"player_{tid}"] = {}
        for joint, vals in joints.items():
            if vals:
                summary[f"player_{tid}"][joint] = {
                    "mean":   round(float(np.mean(vals)), 1),
                    "min":    round(float(np.min(vals)), 1),
                    "max":    round(float(np.max(vals)), 1),
                    "std":    round(float(np.std(vals)), 1),
                    "frames": len(vals),
                }
    return summary


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Badminton AI Coach - pose + tracking")
    parser.add_argument("--input",      required=True)
    parser.add_argument("--output",     default=None)
    parser.add_argument("--display",    action="store_true")
    parser.add_argument("--conf",       type=float, default=0.4)
    parser.add_argument("--max-frames", type=int,   default=None)
    parser.add_argument("--json",       default=None)
    args = parser.parse_args()

    all_frames = process_video(
        input_path=args.input,
        output_path=args.output,
        display=args.display,
        conf=args.conf,
        max_frames=args.max_frames,
    )

    summary = summarise(all_frames)
    print("\n-- Angle summary --")
    print(json.dumps(summary, indent=2))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved -> {args.json}")


if __name__ == "__main__":
    main()