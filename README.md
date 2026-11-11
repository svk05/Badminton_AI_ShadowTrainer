# Badminton AI Shadow Trainer

A shadow-practice coach for badminton. It learns shot patterns from match footage and calls out realistic shot sequences (court zone + shot type) through earphones or a speaker while you do solo footwork.

**Pipeline:** video → pose tracking (YOLOv8-Pose) → manual shot labelling → Markov-chain pattern analysis → timed audio cues on phone.

---

## How it works

1. **Track** – YOLOv8-Pose tracks the player; a manual 4-corner court calibration (homography) maps their position to one of six court zones.
2. **Label** – while the video plays you press SPACE at each shot and a number key for its type. Zone and timing are recorded automatically → `cues.json`.
3. **Analyse** – a first-order Markov chain over (zone, shot) states gives transition probabilities, shot frequencies, n-gram sequences, timing stats and entropy (how predictable the player is).
4. **Train** – the shadow trainer samples rallies from those transitions, with physics-based delays (court distance, shot recovery time), and reads them out.
5. **Compare** – your own cues are compared against a pro's report to produce prioritised coaching tips.

### Court zones

```
 5 back-left  | 6 back-right
 3 mid-left   | 4 mid-right
 1 front-left | 2 front-right
 ------------ net ------------
```

Shot constraints: smash / clear / drop from zones 5–6, lift / net-shot / net-kill from 1–2, drive from 3–4.

---

## Files

| File | Purpose |
|---|---|
| `badminton_tracker.py` | v1 – YOLOv8-Pose player tracking and joint angles |
| `badminton_tracker_v2.py` | v2 – adds heuristic shuttle tracking and court homography |
| `badminton_tracker_v3.py` | v3 – shot labelling tool, outputs `cues.json` |
| `ml_analysis.py` | Markov-chain rally analysis → trainer report |
| `shadow_trainer.py` | Flask web app: serves the shadow coach to your phone over WiFi |
| `Coaching_tips.py` | Compares player cues vs a pro report → HIGH / MED / LOW tips |
| `cues*.json` | Labelled shot data (`KV` = Kunlavat Vitidsarn, `M` = own play, no suffix = Shi Yu Qi) |
| `rally_cues*.json` | Analysis reports generated from the cue files |
| `coaching_report_1.txt` | Example coaching output |
| `Badminton_Shadow_Coach/` | React Native (Expo) Android app – standalone, no PC needed |

> Usage lines inside some scripts still use older filenames (e.g. `analyse_pro_video.py`, `rally_ml.py`, `ai_coach.py`, `compare_player.py`). Use the filenames in this repo.

---

## Setup

```
pip install opencv-python ultralytics numpy flask
```

The YOLOv8-Pose weights download automatically on first run. Videos are not included in the repo.

## Usage

```
# 1. Label shots from a video (click 4 court corners first)
python badminton_tracker_v3.py --input video.mp4 --output cues.json --speed 0.5

# 2. Build the pattern report
python ml_analysis.py --input cues.json --output report.txt --name "Player"

# 3. Run the shadow trainer, then open http://<PC_IP>:5000 on your phone (same WiFi)
python shadow_trainer.py --report rally_cues.json

# 4. Compare yourself against a pro
python Coaching_tips.py --player cuesM.json --pro rally_cues.json --output tips.txt
```

For the Android app, see [`Badminton_Shadow_Coach/README.md`](Badminton_Shadow_Coach/README.md) (Expo + EAS cloud build).

---

## Limitations / next steps

- Shot types are labelled by hand; there is no automatic shot classifier yet.
- Pattern model is a first-order Markov chain. A planned upgrade is a neural sequence model (Transformer/LSTM with player-style embeddings) for next-shot prediction, plus a pose-based shot classifier to automate labelling.
- Shuttle tracking is heuristic (frame differencing), not learned.

---

## AI usage

This project was developed with AI assistance (Claude, Anthropic), which was used for code generation, debugging and documentation. Project design, shot labelling, badminton domain rules and testing are my own.
