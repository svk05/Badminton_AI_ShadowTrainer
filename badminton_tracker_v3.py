"""
Badminton Shadow Coach — Pro Video Analyser v3
===============================================
For single-player training videos.

How it works:
  - Tracks the player automatically and shows their zone on the minimap
  - YOU watch the video and press SPACE whenever a shot is played
  - A menu appears — press a number key to label the shot type
  - Zone is read automatically from player position at that moment
  - Timing between cues is recorded automatically

Controls:
  SPACE       = mark a shot now (then press shot key)
  U           = undo last cue
  Q           = quit and save

Shot keys (shown on screen):
  1 = smash       2 = clear
  3 = drop        4 = net-kill
  5 = net-shot    6 = lift
  7 = drive

Output: cues.json for the Android shadow coach app

Dependencies:
    pip install opencv-python ultralytics numpy

Usage:
    python analyse_pro_video.py --input video.mp4 --output cues.json
    python analyse_pro_video.py --input video.mp4 --output cues.json --speed 0.5
      (--speed 0.5 plays at half speed for easier labelling)
"""

import cv2
import numpy as np
import argparse
import json
import math
import time
from collections import deque
from ultralytics import YOLO


# ─── Constants ────────────────────────────────────────────────────────────────

SHOT_KEYS = {
    ord('1'): "smash",
    ord('2'): "clear",
    ord('3'): "drop",
    ord('4'): "net-kill",
    ord('5'): "net-shot",
    ord('6'): "lift",
    ord('7'): "drive",
}

SHOT_MENU = [
    "1=smash", "2=clear", "3=drop", "4=net-kill",
    "5=net-shot", "6=lift", "7=drive",
]

ZONE_COLS = {
    1:(0,200,255), 2:(0,150,255),
    3:(0,255,150), 4:(0,200,100),
    5:(255,180,0), 6:(255,120,0),
}

KP = {
    "left_shoulder":  5,  "right_shoulder": 6,
    "left_elbow":     7,  "right_elbow":    8,
    "left_wrist":     9,  "right_wrist":    10,
    "left_hip":       11, "right_hip":      12,
    "left_knee":      13, "right_knee":     14,
    "left_ankle":     15, "right_ankle":    16,
}

SKELETON = [
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]


# ─── Court zone ───────────────────────────────────────────────────────────────

def court_zone(nx, ny):
    col = 0 if nx < 0.5 else 1
    row = 0 if ny >= 0.67 else (1 if ny >= 0.33 else 2)
    return [[1,2],[3,4],[5,6]][row][col]


# ─── Court calibrator ─────────────────────────────────────────────────────────

class CourtCalibrator:
    LABELS = ["1-Near-Left", "2-Near-Right", "3-Far-Right", "4-Far-Left"]

    def __init__(self):
        self.points = []
        self.H      = None

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            print(f"  Point {len(self.points)}: ({x}, {y})")

    def calibrate(self, frame):
        win = "Court Calibration"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, frame.shape[1], frame.shape[0])
        cv2.imshow(win, frame)
        cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
        cv2.setMouseCallback(win, self._mouse_cb)

        print("\n=== COURT CALIBRATION ===")
        print("Click: Near-Left, Near-Right, Far-Right, Far-Left")
        print("ENTER to confirm, R to reset\n")

        while True:
            vis = frame.copy()
            for i, pt in enumerate(self.points):
                cv2.circle(vis, pt, 12, (0,255,0), -1)
                cv2.putText(vis, self.LABELS[i], (pt[0]+14, pt[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
            if len(self.points) == 4:
                cv2.polylines(vis, [np.array(self.points, np.int32)],
                              True, (0,200,255), 3)
            msg = (f"Click {4-len(self.points)} more" if len(self.points)<4
                   else "Press ENTER to confirm, R to reset")
            cv2.putText(vis, msg, (20,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,255), 3)
            cv2.putText(vis, f"{len(self.points)}/4", (20,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
            cv2.imshow(win, vis)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 32) and len(self.points) == 4:
                break
            if key == ord('r'):
                self.points = []
                print("Reset.")

        cv2.destroyWindow(win)
        src = np.array(self.points, dtype=np.float32)
        dst = np.array([[0,1],[1,1],[1,0],[0,0]], dtype=np.float32)
        self.H, _ = cv2.findHomography(src, dst)
        print("Calibration done.\n")
        return self.H is not None

    def pixel_to_court(self, px, py):
        if self.H is None:
            return None
        pt  = np.array([[[float(px), float(py)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.H)[0][0]
        return (float(np.clip(out[0],0,1)), float(np.clip(out[1],0,1)))

    def draw_overlay(self, frame):
        if len(self.points) == 4:
            cv2.polylines(frame, [np.array(self.points, np.int32)],
                          True, (0,255,200), 1)


# ─── Zone smoother ────────────────────────────────────────────────────────────

class ZoneSmoother:
    def __init__(self, stability=5):
        self.xs        = deque(maxlen=10)
        self.ys        = deque(maxlen=10)
        self.current   = 3
        self.candidate = 3
        self.count     = 0
        self.stability = stability

    def update(self, court_pos):
        if court_pos is None:
            return self.current
        self.xs.append(court_pos[0])
        self.ys.append(court_pos[1])
        if len(self.xs) < 3:
            return self.current
        zone = court_zone(float(np.median(self.xs)), float(np.median(self.ys)))
        if zone == self.candidate:
            self.count += 1
            if self.count >= self.stability:
                self.current = zone
        else:
            self.candidate = zone
            self.count     = 1
        return self.current


# ─── Player locker ────────────────────────────────────────────────────────────

class PlayerLocker:
    def __init__(self, lock_after=30):
        self.locked_id   = None
        self.candidates  = {}
        self.lock_after  = lock_after
        self.frame_count = 0

    def update(self, boxes, kp_data):
        self.frame_count += 1
        visible = {}
        for i, box in enumerate(boxes):
            if box.id is None:
                continue
            tid = int(box.id[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            visible[tid] = (kp_data[i].cpu().numpy().tolist(),
                            (x1,y1,x2,y2),
                            (x2-x1)*(y2-y1))

        if self.locked_id is None:
            for tid, (_,_,area) in visible.items():
                self.candidates.setdefault(tid,[]).append(area)
            if self.frame_count >= self.lock_after and self.candidates:
                self.locked_id = max(self.candidates,
                                     key=lambda t: np.median(self.candidates[t]))
                print(f"  [PlayerLocker] Locked onto ID {self.locked_id}")

        if self.locked_id in visible:
            kps, bbox, _ = visible[self.locked_id]
            return kps, bbox
        if visible:
            best = max(visible.values(), key=lambda v: v[2])
            return best[0], best[1]
        return None, None


# ─── Drawing helpers ──────────────────────────────────────────────────────────

def draw_skeleton(frame, kps, col):
    for a_i,b_i in SKELETON:
        a,b = kps[a_i], kps[b_i]
        if a[2]<0.25 or b[2]<0.25: continue
        cv2.line(frame,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),col,2)
    for kp in kps:
        if kp[2]<0.25: continue
        cv2.circle(frame,(int(kp[0]),int(kp[1])),3,col,-1)


def draw_zone_badge(frame, zone, px, py):
    col = ZONE_COLS.get(zone,(200,200,200))
    cv2.circle(frame,(px,py),22,col,-1)
    cv2.circle(frame,(px,py),22,(0,0,0),2)
    cv2.putText(frame,str(zone),(px-8,py+8),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),3)


def draw_minimap(frame, court_pos, zone, W, H):
    MW,MH,mg = 110,165,10
    ox,oy = W-MW-mg, mg
    cv2.rectangle(frame,(ox,oy),(ox+MW,oy+MH),(25,35,45),-1)
    cv2.rectangle(frame,(ox,oy),(ox+MW,oy+MH),(80,100,110),1)
    for row in range(3):
        for col in range(2):
            z  = [[1,2],[3,4],[5,6]][row][col]
            zx = ox + col*(MW//2)
            zy = oy + (2-row)*(MH//3)
            zw,zh = MW//2, MH//3
            active = (z == zone)
            cv2.rectangle(frame,(zx,zy),(zx+zw,zy+zh),
                          ZONE_COLS[z] if active else (40,50,60),
                          -1 if active else 1)
            cv2.putText(frame,str(z),(zx+zw//2-8,zy+zh//2+6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,
                        (0,0,0) if active else (100,120,130),1)
    if court_pos:
        cv2.circle(frame,
                   (int(ox+court_pos[0]*MW), int(oy+(1-court_pos[1])*MH)),
                   6,(0,255,255),-1)


def draw_shot_menu(frame, W, H):
    """Overlay the shot selection menu when waiting for input."""
    # Dark panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, H-220), (W, H), (10,10,10), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "SELECT SHOT:", (20, H-185),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,100), 2)

    # Two rows of 4 + 3
    row1 = SHOT_MENU[:4]
    row2 = SHOT_MENU[4:]
    for i, label in enumerate(row1):
        cv2.putText(frame, label, (20 + i*190, H-140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,220,80), 2)
    for i, label in enumerate(row2):
        cv2.putText(frame, label, (20 + i*190, H-95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,220,80), 2)

    cv2.putText(frame, "ESC = cancel", (20, H-50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)


def draw_hud(frame, frame_idx, total, fps, cues, zone, waiting, locked):
    W = frame.shape[1]

    # Top bar
    t_str    = f"{frame_idx/fps:.1f}s"
    lock_str = f"ID:{locked}" if locked else "locking..."
    cue_str  = f"Cues:{len(cues)}"
    zone_str = f"Zone:{zone}"
    cv2.putText(frame,
                f"Frame {frame_idx:05d}/{total} | {t_str} | {lock_str} | {cue_str} | {zone_str}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)

    if not waiting:
        # Controls reminder at bottom
        cv2.putText(frame, "SPACE=mark shot  U=undo  Q=quit+save",
                    (10, frame.shape[0]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160,160,160), 1)

    # Last cue
    if cues:
        last = cues[-1]
        cv2.putText(frame,
                    f"Last: {last['zone']} {last['shot']}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,100), 2)


def draw_flash(frame, text, col=(0,255,100)):
    H,W = frame.shape[:2]
    cv2.putText(frame, text, (W//2-len(text)*18, H//2),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, col, 5)


# ─── Main ─────────────────────────────────────────────────────────────────────

def analyse(input_path, output_path, speed):

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

    ret, first_frame = cap.read()
    if not ret:
        raise IOError("Cannot read first frame.")

    print("[3/3] Court calibration ...")
    cal = CourtCalibrator()
    cal.calibrate(first_frame)

    locker       = PlayerLocker(lock_after=int(fps*1.5))
    zone_smoother= ZoneSmoother(stability=4)

    cues          = []
    last_cue_t    = None
    frame_idx     = 0
    frames_buf    = [first_frame]

    # State
    waiting_shot  = False   # True = paused, showing shot menu
    pending_zone  = 3
    pending_t     = 0.0
    flash_text    = ""
    flash_until   = -1
    flash_col     = (0,255,100)

    # Playback delay based on speed factor
    frame_delay   = max(1, int((1000 / fps) / speed))

    win = "Badminton Analyser — SPACE=mark  U=undo  Q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(W, 1280), min(H, 720))

    print("\n=== READY ===")
    print("Watch the video. Press SPACE when a shot is played.")
    print("Then press 1-7 to label the shot type.\n")

    court_pos = None
    zone      = 3

    while True:
        if frames_buf:
            frame = frames_buf.pop(0)
        else:
            ret, frame = cap.read()
            if not ret:
                break

        t = frame_idx / fps

        # ── Pose tracking ────────────────────────────────────────────────
        results = model.track(frame, persist=True, conf=0.4, verbose=False)
        kps, bbox = None, None

        if results and results[0].keypoints is not None:
            boxes   = results[0].boxes
            kp_data = results[0].keypoints.data
            kps, bbox = locker.update(boxes, kp_data)

        if kps is not None and bbox is not None:
            x1,y1,x2,y2 = bbox
            la,ra = kps[KP["left_ankle"]], kps[KP["right_ankle"]]
            if la[2]>0.25 and ra[2]>0.25:
                fx,fy = (la[0]+ra[0])/2,(la[1]+ra[1])/2
            elif la[2]>0.25:
                fx,fy = la[0],la[1]
            elif ra[2]>0.25:
                fx,fy = ra[0],ra[1]
            else:
                fx,fy = (x1+x2)/2, float(y2)

            court_pos = cal.pixel_to_court(fx, fy)
            zone      = zone_smoother.update(court_pos)

            col = ZONE_COLS.get(zone,(0,200,255))
            cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
            draw_skeleton(frame, kps, col)
            draw_zone_badge(frame, zone, int(fx), max(int(fy)-40,20))

        cal.draw_overlay(frame)
        draw_minimap(frame, court_pos, zone, W, H)

        # ── Flash overlay ─────────────────────────────────────────────────
        if frame_idx <= flash_until:
            draw_flash(frame, flash_text, flash_col)

        # ── Shot menu overlay ─────────────────────────────────────────────
        if waiting_shot:
            draw_shot_menu(frame, W, H)

        draw_hud(frame, frame_idx, total, fps, cues, zone, waiting_shot,
                 locker.locked_id)

        cv2.imshow(win, frame)

        # ── Key handling ──────────────────────────────────────────────────
        if waiting_shot:
            # Paused — wait for shot key or ESC
            key = cv2.waitKey(0) & 0xFF

            if key == 27:   # ESC = cancel
                waiting_shot = False
                print("  Cancelled.")

            elif key in SHOT_KEYS:
                shot  = SHOT_KEYS[key]
                delay = (pending_t - last_cue_t) if last_cue_t is not None else 1.0
                delay = round(max(0.2, min(delay, 8.0)), 2)
                cues.append({"zone": pending_zone, "shot": shot, "delay_s": delay})
                last_cue_t   = pending_t
                waiting_shot = False
                flash_text   = f"{pending_zone}  {shot}"
                flash_until  = frame_idx + int(fps * 0.8)
                flash_col    = (0,255,100)
                print(f"  Cue {len(cues):03d}: zone={pending_zone}  "
                      f"shot={shot}  delay={delay:.2f}s  t={pending_t:.2f}s")

        else:
            # Normal playback
            key = cv2.waitKey(frame_delay) & 0xFF

            if key == ord('q'):
                print("\nStopping early and saving ...")
                break

            elif key == ord(' '):
                # Mark shot at current frame
                waiting_shot  = True
                pending_zone  = zone
                pending_t     = t
                print(f"  [SPACE] t={t:.2f}s  zone={zone} — select shot ...")

            elif key == ord('u'):
                # Undo last cue
                if cues:
                    removed = cues.pop()
                    last_cue_t  = cues[-1]["delay_s"] if cues else None
                    flash_text  = f"UNDO: {removed['zone']} {removed['shot']}"
                    flash_until = frame_idx + int(fps * 1.0)
                    flash_col   = (0,100,255)
                    print(f"  Undid: {removed}")
                else:
                    print("  Nothing to undo.")

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # ── Save JSON ─────────────────────────────────────────────────────────────
    delays = [c["delay_s"] for c in cues]
    output = {
        "source":           input_path.split("\\")[-1].split("/")[-1],
        "total_cues":       len(cues),
        "avg_rally_pace_s": round(float(np.mean(delays)),3) if delays else 1.0,
        "min_delay_s":      round(float(np.min(delays)),3)  if delays else 0.3,
        "max_delay_s":      round(float(np.max(delays)),3)  if delays else 5.0,
        "zone_layout": {
            "1":"front-left","2":"front-right",
            "3":"mid-left",  "4":"mid-right",
            "5":"back-left", "6":"back-right",
        },
        "cues": cues,
    }

    with open(output_path,"w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== SAVED ===")
    print(f"  Cues  : {len(cues)}")
    print(f"  Pace  : {output['avg_rally_pace_s']:.2f}s avg")
    print(f"  File  : {output_path}")
    if cues:
        from collections import Counter
        shot_counts = Counter(c["shot"] for c in cues)
        zone_counts = Counter(c["zone"] for c in cues)
        print(f"  Shots : {dict(shot_counts)}")
        print(f"  Zones : {dict(sorted(zone_counts.items()))}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   required=True)
    parser.add_argument("--output",  default="cues.json")
    parser.add_argument("--speed",   type=float, default=1.0,
                        help="Playback speed (0.5=half speed, 1.0=normal)")
    args = parser.parse_args()
    analyse(args.input, args.output, args.speed)

if __name__ == "__main__":
    main()