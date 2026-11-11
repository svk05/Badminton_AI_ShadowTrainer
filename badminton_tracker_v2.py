"""
Badminton AI Coach — Week 3-4: Shuttle Tracking + Court Homography
==================================================================
Builds on Week 1-2 (YOLOv8-Pose player tracking).
New additions:
  - Heuristic shuttle tracker (frame-diff + white blob detection)
  - Manual court calibration (click 4 corners on first frame)
  - Homography: maps pixel coords -> normalised court coords [0,1]
  - Trail visualisation for shuttle and player feet

Dependencies:
    pip install opencv-python ultralytics numpy

Usage:
    python badminton_tracker_v2.py --input match.mp4 --output out.mp4 --display
    python badminton_tracker_v2.py --input match.mp4 --output out.mp4 --json stats.json

Court calibration:
    On startup a frame will appear. Click the 4 court corners IN ORDER:
        1. Near-left   (bottom-left from your camera view)
        2. Near-right  (bottom-right)
        3. Far-right   (top-right)
        4. Far-left    (top-left)
    Press ENTER or SPACE to confirm. Press R to reset clicks.
"""

import cv2
import numpy as np
import argparse
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import deque
from ultralytics import YOLO


# ─── Court definition (real-world normalised coords) ──────────────────────────
# We map the 4 clicked corners to a standard [0,1] x [0,1] court rectangle.
# x=0 left, x=1 right, y=0 near (camera side), y=1 far
COURT_DST = np.array([
    [0.0, 1.0],  # near-left
    [1.0, 1.0],  # near-right
    [1.0, 0.0],  # far-right
    [0.0, 0.0],  # far-left
], dtype=np.float32)

TRAIL_LEN = 30   # frames to keep in shuttle/player trail


# ─── COCO keypoint indices ─────────────────────────────────────────────────────
KP = {
    "nose": 0,
    "left_shoulder": 5,  "right_shoulder": 6,
    "left_elbow": 7,     "right_elbow": 8,
    "left_wrist": 9,     "right_wrist": 10,
    "left_hip": 11,      "right_hip": 12,
    "left_knee": 13,     "right_knee": 14,
    "left_ankle": 15,    "right_ankle": 16,
}

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
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
    track_id:       int
    bbox:           tuple
    center:         tuple
    court_pos:      Optional[tuple]   # normalised (x,y) on court
    keypoints:      list
    angles:         JointAngles
    frame_idx:      int

@dataclass
class ShuttleDetection:
    pixel_pos:  Optional[tuple]       # (x, y) in frame
    court_pos:  Optional[tuple]       # normalised (x, y) on court
    confidence: float
    frame_idx:  int

@dataclass
class FrameData:
    frame_idx: int
    players:   list = field(default_factory=list)
    shuttle:   Optional[ShuttleDetection] = None


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def angle_between(a, b, c) -> float:
    ba = (a[0]-b[0], a[1]-b[1])
    bc = (c[0]-b[0], c[1]-b[1])
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag = math.hypot(*ba) * math.hypot(*bc)
    if mag < 1e-6:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot/mag))))

def extract_angles(kps, conf_thresh=0.3) -> JointAngles:
    def pt(name):
        k = kps[KP[name]]; return (k[0], k[1]), k[2]
    def safe(a, b, c):
        (pa,ca),(pb,cb),(pc,cc) = pt(a),pt(b),pt(c)
        if ca<conf_thresh or cb<conf_thresh or cc<conf_thresh: return None
        return angle_between(pa, pb, pc)
    return JointAngles(
        right_elbow    = safe("right_shoulder","right_elbow",   "right_wrist"),
        left_elbow     = safe("left_shoulder", "left_elbow",    "left_wrist"),
        right_knee     = safe("right_hip",     "right_knee",    "right_ankle"),
        left_knee      = safe("left_hip",      "left_knee",     "left_ankle"),
        right_shoulder = safe("right_elbow",   "right_shoulder","right_hip"),
        left_shoulder  = safe("left_elbow",    "left_shoulder", "left_hip"),
        right_hip      = safe("right_shoulder","right_hip",     "right_knee"),
        left_hip       = safe("left_shoulder", "left_hip",      "left_knee"),
    )


# ─── Court calibration (manual click) ─────────────────────────────────────────

class CourtCalibrator:
    """Click 4 court corners to compute a homography matrix."""

    LABELS = ["1-Near-Left", "2-Near-Right", "3-Far-Right", "4-Far-Left"]

    def __init__(self):
        self.points = []
        self.H      = None     # pixel -> normalised court (3x3)
        self.H_inv  = None     # normalised court -> pixel

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))

    def calibrate(self, frame):
        """Show frame, let user click 4 corners, compute homography."""
        win = "Court Calibration — click 4 corners (see order in terminal)"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, self._mouse_cb)

        print("\n=== COURT CALIBRATION ===")
        print("Click corners IN THIS ORDER:")
        for lbl in self.LABELS:
            print(f"  {lbl}")
        print("Press ENTER/SPACE to confirm, R to reset.\n")

        while True:
            vis = frame.copy()
            for i, pt in enumerate(self.points):
                cv2.circle(vis, pt, 8, (0, 255, 0), -1)
                cv2.putText(vis, self.LABELS[i], (pt[0]+10, pt[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            if len(self.points) >= 2:
                for i in range(len(self.points)-1):
                    cv2.line(vis, self.points[i], self.points[i+1], (0,200,255), 2)
            if len(self.points) == 4:
                cv2.line(vis, self.points[3], self.points[0], (0,200,255), 2)

            remaining = 4 - len(self.points)
            msg = (f"Click {remaining} more corner(s)" if remaining
                   else "Press ENTER to confirm, R to reset")
            cv2.putText(vis, msg, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow(win, vis)

            key = cv2.waitKey(20) & 0xFF
            if key in (13, 32) and len(self.points) == 4:   # ENTER or SPACE
                break
            if key == ord('r'):
                self.points = []

        cv2.destroyWindow(win)
        self._compute_homography()
        return self.H is not None

    def _compute_homography(self):
        if len(self.points) != 4:
            return
        src = np.array(self.points, dtype=np.float32)
        dst = COURT_DST * np.array([500, 500], dtype=np.float32)  # scale to 500x500
        self.H,     _ = cv2.findHomography(src, dst)
        self.H_inv, _ = cv2.findHomography(dst, src)
        print("Homography computed successfully.")

    def pixel_to_court(self, px, py) -> Optional[tuple]:
        """Map pixel (px,py) -> normalised court (x,y) in [0,1]."""
        if self.H is None:
            return None
        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.H)[0][0]
        return (float(out[0]) / 500.0, float(out[1]) / 500.0)

    def draw_court_overlay(self, frame):
        """Draw the court boundary on the frame."""
        if len(self.points) != 4:
            return
        pts = np.array(self.points, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True,
                      color=(0, 255, 200), thickness=2)
        for i, pt in enumerate(self.points):
            cv2.circle(frame, pt, 6, (0,255,200), -1)


# ─── Shuttle tracker ──────────────────────────────────────────────────────────

class ShuttleTracker:
    """
    Heuristic shuttle detector using frame differencing + white blob search.
    Works best with a fixed camera.
    Steps per frame:
      1. Frame-diff to find moving regions
      2. Within moving regions, look for small bright (white/yellow) blobs
      3. Filter by size (shuttle is small, ~5-25px diameter)
      4. Pick the candidate closest to the predicted position (Kalman-lite)
    """

    def __init__(self):
        self.prev_gray   = None
        self.last_pos    = None
        self.velocity    = (0, 0)
        self.lost_frames = 0
        self.MAX_LOST    = 10

    def detect(self, frame) -> Optional[tuple]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        # ── Frame difference ────────────────────────────────────────────
        diff = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray
        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.dilate(motion_mask, None, iterations=2)

        # ── White/bright blob detection ─────────────────────────────────
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Shuttle: low saturation (white feathers) OR yellowish cork
        white_mask  = cv2.inRange(hsv, (0,  0, 180), (180, 60, 255))
        yellow_mask = cv2.inRange(hsv, (15, 80, 150), (35, 255, 255))
        shuttle_mask = cv2.bitwise_or(white_mask, yellow_mask)

        # Combine with motion
        combined = cv2.bitwise_and(shuttle_mask, motion_mask)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,
                                    np.ones((3,3), np.uint8))

        # ── Find contours ────────────────────────────────────────────────
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (4 < area < 800):          # shuttle is small
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if radius > 25:                   # too large
                continue
            circularity = 0.0
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * math.pi * area / (perimeter ** 2)
            if circularity < 0.3:             # too irregular
                continue
            candidates.append((int(cx), int(cy), circularity, area))

        if not candidates:
            self.lost_frames += 1
            if self.lost_frames > self.MAX_LOST:
                self.last_pos = None
            return None

        # ── Pick best candidate ──────────────────────────────────────────
        if self.last_pos is not None:
            # Predict next position using last velocity
            pred_x = self.last_pos[0] + self.velocity[0]
            pred_y = self.last_pos[1] + self.velocity[1]
            # Score: balance distance from prediction vs circularity
            def score(c):
                dist = math.hypot(c[0]-pred_x, c[1]-pred_y)
                return dist - c[2]*20   # lower is better
            candidates.sort(key=score)
        else:
            # No prior: pick most circular
            candidates.sort(key=lambda c: -c[2])

        best = candidates[0]
        new_pos = (best[0], best[1])

        if self.last_pos is not None:
            self.velocity = (
                int(0.7*(new_pos[0]-self.last_pos[0]) + 0.3*self.velocity[0]),
                int(0.7*(new_pos[1]-self.last_pos[1]) + 0.3*self.velocity[1]),
            )
        self.last_pos    = new_pos
        self.lost_frames = 0
        return new_pos


# ─── Visualisation ────────────────────────────────────────────────────────────

COLOURS = [(0,210,255),(255,100,40),(80,255,120),(255,60,200)]

def colour_for(tid): return COLOURS[int(tid) % len(COLOURS)]

def draw_player(frame, player: PlayerFrame):
    col = colour_for(player.track_id)
    x1,y1,x2,y2 = player.bbox
    cv2.rectangle(frame, (x1,y1),(x2,y2), col, 2)
    label = f"P{player.track_id}"
    if player.court_pos:
        label += f" ({player.court_pos[0]:.2f},{player.court_pos[1]:.2f})"
    cv2.putText(frame, label, (x1, max(y1-8,12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

    for a_i,b_i in SKELETON:
        a,b = player.keypoints[a_i], player.keypoints[b_i]
        if a[2]<0.3 or b[2]<0.3: continue
        cv2.line(frame,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),col,2)
    for kp in player.keypoints:
        if kp[2]<0.3: continue
        cv2.circle(frame,(int(kp[0]),int(kp[1])),3,col,-1)

    ang = player.angles
    lines = []
    if ang.right_elbow  is not None: lines.append(f"RE {ang.right_elbow:.0f}")
    if ang.left_elbow   is not None: lines.append(f"LE {ang.left_elbow:.0f}")
    if ang.right_knee   is not None: lines.append(f"RK {ang.right_knee:.0f}")
    if ang.left_knee    is not None: lines.append(f"LK {ang.left_knee:.0f}")
    for i,ln in enumerate(lines):
        cv2.putText(frame, ln, (x1, y2+14+i*14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

def draw_shuttle(frame, shuttle: ShuttleDetection,
                 trail: deque, calibrator: CourtCalibrator):
    # Trail
    pts = list(trail)
    for i in range(1, len(pts)):
        if pts[i-1] and pts[i]:
            alpha = i / len(pts)
            col = (int(255*alpha), int(255*(1-alpha)), 80)
            cv2.line(frame, pts[i-1], pts[i], col, 2)

    if shuttle.pixel_pos:
        cx, cy = shuttle.pixel_pos
        cv2.circle(frame, (cx,cy), 8, (0,255,255), 2)
        cv2.circle(frame, (cx,cy), 3, (0,255,255), -1)
        if shuttle.court_pos:
            cv2.putText(frame,
                        f"S ({shuttle.court_pos[0]:.2f},{shuttle.court_pos[1]:.2f})",
                        (cx+10, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)

def draw_minimap(frame, all_players, shuttle_pos, frame_w, frame_h):
    """Draw a small top-down court minimap in the corner."""
    MAP_W, MAP_H = 120, 180
    margin = 10
    ox = frame_w - MAP_W - margin
    oy = margin

    # Background
    cv2.rectangle(frame, (ox,oy),(ox+MAP_W,oy+MAP_H),(30,30,30),-1)
    cv2.rectangle(frame, (ox,oy),(ox+MAP_W,oy+MAP_H),(100,100,100),1)

    # Court lines (simplified)
    mid_y = oy + MAP_H//2
    cv2.line(frame, (ox, mid_y),(ox+MAP_W, mid_y),(80,80,80),1)
    cv2.line(frame, (ox+MAP_W//4, oy),(ox+MAP_W//4, oy+MAP_H),(60,60,60),1)
    cv2.line(frame, (ox+3*MAP_W//4, oy),(ox+3*MAP_W//4, oy+MAP_H),(60,60,60),1)

    def court_to_map(cx, cy):
        return (int(ox + cx*MAP_W), int(oy + (1-cy)*MAP_H))

    # Players
    for pid, pos in all_players:
        if pos:
            mx, my = court_to_map(pos[0], pos[1])
            col = colour_for(pid)
            cv2.circle(frame, (mx,my), 6, col, -1)
            cv2.putText(frame, str(pid),(mx+4,my-4),
                        cv2.FONT_HERSHEY_SIMPLEX,0.35,col,1)

    # Shuttle
    if shuttle_pos:
        mx, my = court_to_map(shuttle_pos[0], shuttle_pos[1])
        cv2.circle(frame,(mx,my),4,(0,255,255),-1)

    cv2.putText(frame,"Court",(ox+4,oy+MAP_H-5),
                cv2.FONT_HERSHEY_SIMPLEX,0.35,(150,150,150),1)

def draw_hud(frame, frame_idx, n_players, shuttle_detected):
    shuttle_str = "YES" if shuttle_detected else "NO"
    cv2.putText(frame,
                f"Frame {frame_idx:05d} | Players: {n_players} | Shuttle: {shuttle_str}",
                (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)


# ─── Main pipeline ────────────────────────────────────────────────────────────

def process_video(input_path, output_path, display, conf=0.4, max_frames=None):
    print("[1/3] Loading YOLOv8-Pose ...")
    model = YOLO("yolov8n-pose.pt")

    print("[2/3] Opening video ...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {input_path}")

    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"    {W}x{H}  {fps:.1f}fps  {total} frames")

    # Read first frame for calibration
    ret, first_frame = cap.read()
    if not ret:
        raise IOError("Could not read first frame.")

    print("[3/3] Court calibration ...")
    calibrator = CourtCalibrator()
    calibrated = calibrator.calibrate(first_frame)
    if not calibrated:
        print("  Calibration skipped — court positions won't be available.")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    shuttle_tracker = ShuttleTracker()
    shuttle_trail   = deque(maxlen=TRAIL_LEN)
    all_frames      = []
    frame_idx       = 0

    # Process first frame too
    frames_to_process = [first_frame]

    while True:
        if frames_to_process:
            frame = frames_to_process.pop(0)
        else:
            ret, frame = cap.read()
            if not ret:
                break
        if max_frames and frame_idx >= max_frames:
            break

        fd = FrameData(frame_idx=frame_idx)

        # ── Court overlay ────────────────────────────────────────────────
        calibrator.draw_court_overlay(frame)

        # ── Player detection + pose ──────────────────────────────────────
        results = model.track(frame, persist=True, conf=conf, verbose=False)
        player_court_positions = []

        if results and results[0].keypoints is not None:
            boxes   = results[0].boxes
            kp_data = results[0].keypoints.data

            for i, box in enumerate(boxes):
                if box.id is None: continue
                tid = int(box.id[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                kps    = kp_data[i].cpu().numpy().tolist()
                angles = extract_angles(kps)
                cx,cy  = (x1+x2)//2, (y1+y2)//2

                # Foot position for court mapping (average of ankles if visible)
                foot_x, foot_y = cx, y2
                la, ra = kps[KP["left_ankle"]], kps[KP["right_ankle"]]
                if la[2]>0.3 and ra[2]>0.3:
                    foot_x = int((la[0]+ra[0])/2)
                    foot_y = int((la[1]+ra[1])/2)
                elif la[2]>0.3:
                    foot_x,foot_y = int(la[0]),int(la[1])
                elif ra[2]>0.3:
                    foot_x,foot_y = int(ra[0]),int(ra[1])

                court_pos = calibrator.pixel_to_court(foot_x, foot_y)

                pf = PlayerFrame(
                    track_id=tid, bbox=(x1,y1,x2,y2),
                    center=(cx,cy), court_pos=court_pos,
                    keypoints=kps, angles=angles, frame_idx=frame_idx,
                )
                fd.players.append(pf)
                draw_player(frame, pf)
                player_court_positions.append((tid, court_pos))

        # ── Shuttle tracking ─────────────────────────────────────────────
        shuttle_px = shuttle_tracker.detect(frame)
        shuttle_court = calibrator.pixel_to_court(*shuttle_px) if shuttle_px else None
        shuttle_det = ShuttleDetection(
            pixel_pos=shuttle_px,
            court_pos=shuttle_court,
            confidence=1.0 if shuttle_px else 0.0,
            frame_idx=frame_idx,
        )
        fd.shuttle = shuttle_det
        shuttle_trail.append(shuttle_px)
        draw_shuttle(frame, shuttle_det, shuttle_trail, calibrator)

        # ── Minimap ──────────────────────────────────────────────────────
        if calibrated:
            draw_minimap(frame, player_court_positions, shuttle_court, W, H)

        draw_hud(frame, frame_idx, len(fd.players), shuttle_px is not None)
        all_frames.append(fd)

        if writer: writer.write(frame)
        if display:
            cv2.imshow("Badminton Tracker v2", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  ... {frame_idx}/{total} frames")

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()

    unique_ids = {p.track_id for fd in all_frames for p in fd.players}
    shuttle_frames = sum(1 for fd in all_frames if fd.shuttle and fd.shuttle.pixel_pos)
    print(f"\nDone - {frame_idx} frames | {len(unique_ids)} players | "
          f"Shuttle detected in {shuttle_frames} frames ({100*shuttle_frames/max(frame_idx,1):.1f}%)")
    return all_frames


# ─── Analytics summary ────────────────────────────────────────────────────────

def summarise(all_frames):
    stats = {}
    shuttle_positions = []

    for fd in all_frames:
        for p in fd.players:
            tid = p.track_id
            if tid not in stats:
                stats[tid] = {j: [] for j in JointAngles.__dataclass_fields__}
                stats[tid]["court_x"] = []
                stats[tid]["court_y"] = []
            for joint, val in asdict(p.angles).items():
                if val is not None:
                    stats[tid][joint].append(val)
            if p.court_pos:
                stats[tid]["court_x"].append(p.court_pos[0])
                stats[tid]["court_y"].append(p.court_pos[1])

        if fd.shuttle and fd.shuttle.court_pos:
            shuttle_positions.append(fd.shuttle.court_pos)

    summary = {"players": {}, "shuttle": {}}
    for tid, data in stats.items():
        summary["players"][f"player_{tid}"] = {}
        for key, vals in data.items():
            if vals:
                summary["players"][f"player_{tid}"][key] = {
                    "mean": round(float(np.mean(vals)), 3),
                    "min":  round(float(np.min(vals)),  3),
                    "max":  round(float(np.max(vals)),  3),
                    "std":  round(float(np.std(vals)),  3),
                    "n":    len(vals),
                }

    if shuttle_positions:
        xs = [p[0] for p in shuttle_positions]
        ys = [p[1] for p in shuttle_positions]
        summary["shuttle"] = {
            "detected_frames": len(shuttle_positions),
            "court_x": {"mean": round(float(np.mean(xs)),3),
                        "min":  round(float(np.min(xs)), 3),
                        "max":  round(float(np.max(xs)), 3)},
            "court_y": {"mean": round(float(np.mean(ys)),3),
                        "min":  round(float(np.min(ys)), 3),
                        "max":  round(float(np.max(ys)), 3)},
        }
    return summary


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Badminton Tracker v2 - shuttle + court")
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
    print("\n-- Summary --")
    print(json.dumps(summary, indent=2))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved -> {args.json}")


if __name__ == "__main__":
    main()