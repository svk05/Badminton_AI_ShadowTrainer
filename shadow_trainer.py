"""
Badminton AI Coach
==================
Physics-aware rally generator. Timing accounts for:
  - Court distance (back->front takes longer than front->front)
  - Shot recovery time (drops/clears = slow, smashes = fast)
  - Specific combo speeds (post-smash net-shot = 1.3s, back-court avg = 2.33s)
  - ML transition probabilities from the trainer report

Accepts:  structured JSON report  OR  plain text report  OR  JSON with 'report' key

Install:  pip install flask
Run:      python ai_coach.py --report rally_cues.json
Phone:    http://<PC_IP>:5000  (same WiFi)
"""

from flask import Flask, render_template_string, jsonify, request
import json, argparse, socket, random, re, math
from collections import defaultdict

app   = Flask(__name__)
MODEL = {}


# ─── Court physics ────────────────────────────────────────────────────────────
#
# Zone layout:
#   5 (back-left)  | 6 (back-right)     row 2 (far)
#   3 (mid-left)   | 4 (mid-right)      row 1
#   1 (front-left) | 2 (front-right)    row 0 (net)
#
# Key rules (from user specification):
#   - Back->front (5/6 -> 1/2): avg 2.33s, post-smash max ~1.8s
#   - Post-smash net-shot: ~1.3s  (fastest combo)
#   - Drops/clears/lifts: slow shots, more recovery to centre
#   - Smashes: faster recovery needed (opponent's reply is fast)
#   - Net-shots/drives: versatile, use ML avg timings

ZONE_ROW = {1:0, 2:0, 3:1, 4:1, 5:2, 6:2}
ZONE_COL = {1:0, 2:1, 3:0, 4:1, 5:0, 6:1}

# Base recovery time after playing each shot (time to return to centre)
# Longer = player needs more time, so next cue comes later
SHOT_RECOVERY = {
    "smash":    0.0,   # smash is fast; opponent's return is fast too
    "drop":     0.55,  # drop = lunge forward, slow recovery to centre
    "clear":    0.50,  # clear = full swing, slower recovery
    "lift":     0.40,  # lift from low position, moderate recovery
    "net-shot": 0.10,  # already at net, minimal recovery
    "net-kill": 0.05,  # already at net, very fast
    "drive":    0.15,  # mid-court, quick recovery
}

# Court travel time matrix: from zone -> to zone (base seconds)
# Represents time for player to travel between zones
def travel_time(from_zone, to_zone):
    r1, r2 = ZONE_ROW[from_zone], ZONE_ROW[to_zone]
    c1, c2 = ZONE_COL[from_zone], ZONE_COL[to_zone]
    row_diff = abs(r2 - r1)
    col_diff = abs(c2 - c1)
    # Base travel: 0.3s per row, 0.2s per column
    return row_diff * 0.30 + col_diff * 0.20


# ─── Timing engine ────────────────────────────────────────────────────────────

def compute_delay(prev_shot, prev_zone, next_shot, next_zone,
                  pace_factor, ml_avg):
    """
    Compute realistic delay between shots based on:
      1. Court travel distance
      2. Recovery time from previous shot
      3. Specific physics rules (back->front, post-smash net, etc.)
      4. ML average as baseline
      5. Pace factor (warmup/moderate/fast)

    Returns delay in seconds.
    """

    # ── Special combo rules (highest priority) ──────────────────────────────
    # Post-smash -> net-shot: very fast, opponent scrambled to lift
    if prev_shot == "smash" and next_shot in ("net-shot", "net-kill"):
        base = random.gauss(1.30, 0.12)
        return round(max(0.9, base) * pace_factor, 2)

    # Post-smash -> any front zone: max ~1.8s (opponent lifts from scramble)
    if prev_shot == "smash" and next_zone in (1, 2):
        base = random.gauss(1.60, 0.18)
        return round(max(1.1, min(base, 2.0)) * pace_factor, 2)

    # Back->front (not post-smash): avg 2.33s
    if prev_zone in (5, 6) and next_zone in (1, 2) and prev_shot != "smash":
        base = random.gauss(2.33, 0.25)
        return round(max(1.6, min(base, 3.2)) * pace_factor, 2)

    # Front->back (lift to back court): player sprints back
    if prev_zone in (1, 2) and next_zone in (5, 6):
        base = random.gauss(2.10, 0.22)
        return round(max(1.5, min(base, 3.0)) * pace_factor, 2)

    # ── Zone-same or adjacent (use ML avg + recovery + travel) ─────────────
    recovery  = SHOT_RECOVERY.get(prev_shot, 0.25)
    travel    = travel_time(prev_zone, next_zone)

    # Use ML average for the next shot type as baseline
    base = ml_avg if ml_avg else 2.0

    # Add recovery and travel, scaled
    computed = base * 0.6 + recovery * 0.8 + travel * 1.2

    # Jitter ±15%
    jitter = random.gauss(0, computed * 0.12)
    delay  = computed + jitter

    # Shot-specific clamps
    if next_shot in ("drop", "clear"):
        delay = max(delay, 1.8)   # slow shots always have wind-up time
    elif next_shot == "smash":
        delay = max(delay, 1.4)
    elif next_shot in ("net-shot", "net-kill"):
        delay = min(delay, 2.2)   # net play is always snappy

    delay = max(0.5, min(delay, 6.0))
    return round(delay * pace_factor, 2)


# ─── Report parsing ───────────────────────────────────────────────────────────

def parse_json_report(data):
    parsed = {
        "transitions": defaultdict(dict),
        "shot_ml_avg":  {},   # shot -> ML avg seconds
        "zone_shots":   {},
        "avg_delay":    data["timing_overview"]["average_delay_seconds"],
        "min_delay":    data["timing_overview"]["fastest_exchange_seconds"],
        "max_delay":    data["timing_overview"]["slowest_exchange_seconds"],
        "std_delay":    data["timing_overview"]["std_deviation_seconds"],
        "source":       data.get("source", "Unknown"),
    }
    for s in data["shot_frequency"]:
        parsed["shot_ml_avg"][s["shot"]] = s["avg_seconds"]
    for z in data["shots_by_zone"]:
        parsed["zone_shots"][z["zone"]] = {s["shot"]: s["count"] for s in z["shots"]}
    for t in data["top_shot_transitions"]:
        fz, fs = t["from"].split("-", 1)
        tz, ts = t["to"].split("-", 1)
        parsed["transitions"][(int(fz), fs)][(int(tz), ts)] = \
            t["probability_percentage"] / 100.0
    return parsed


def parse_text_report(text):
    text = text.replace("\r", "")
    parsed = {
        "transitions": defaultdict(dict),
        "shot_ml_avg":  {},
        "zone_shots":   {},
        "avg_delay": 2.41, "min_delay": 1.01,
        "max_delay": 7.09, "std_delay": 1.29,
        "source": "Unknown",
    }
    def find(pattern, cast=float):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None

    parsed["source"]    = find(r"Source\s*:\s*(.+)", str) or "Unknown"
    parsed["avg_delay"] = find(r"Average delay between shots\s*:\s*([\d.]+)") or 2.41
    parsed["min_delay"] = find(r"Fastest exchange\s*:\s*([\d.]+)") or 1.01
    parsed["max_delay"] = find(r"Slowest exchange\s*:\s*([\d.]+)") or 7.09
    parsed["std_delay"] = find(r"Std deviation\s*:\s*([\d.]+)") or 1.29

    for m in re.finditer(r"([\w-]+)\s+\d+x\s+[\d.]+%\s+avg\s+([\d.]+)s", text):
        parsed["shot_ml_avg"][m.group(1)] = float(m.group(2))

    for m in re.finditer(r"Zone (\d) \([^)]+\): (.+)", text):
        zone, rest = int(m.group(1)), m.group(2)
        shots = {sm.group(1): int(sm.group(2))
                 for sm in re.finditer(r"([\w-]+)\s+(\d+)x\s+\((\d+)%\)", rest)}
        if shots:
            parsed["zone_shots"][zone] = shots

    in_trans = False
    for line in text.split("\n"):
        if "TOP SHOT TRANSITIONS" in line: in_trans = True; continue
        if in_trans and ("MOST COMMON" in line or "STATE PRED" in line):
            in_trans = False; continue
        if in_trans:
            m = re.match(r"\s+(\d)-([\w-]+)\s+(\d)-([\w-]+)\s+\d+x\s+(\d+)%", line)
            if m:
                parsed["transitions"][(int(m.group(1)), m.group(2))] \
                                      [(int(m.group(3)), m.group(4))] = \
                                      int(m.group(5)) / 100.0
    return parsed


def load_report(path):
    with open(path, encoding="utf-8") as f:
        raw = f.read().replace("\r", "")
    try:
        data = json.loads(raw)
        if "shot_frequency" in data and "shots_by_zone" in data:
            return parse_json_report(data)
        if "report" in data:
            return parse_text_report(data["report"])
    except (json.JSONDecodeError, ValueError):
        pass
    return parse_text_report(raw)


# ─── Model builder ────────────────────────────────────────────────────────────

def build_model(parsed):
    # Normalise transition probabilities
    trans_norm = {}
    for from_s, nexts in parsed["transitions"].items():
        total = sum(nexts.values())
        if total > 0:
            trans_norm[from_s] = {s: v/total for s, v in nexts.items()}

    zone_weights = {z: sum(shots.values())
                    for z, shots in parsed["zone_shots"].items()}

    return {
        "transitions":  trans_norm,
        "zone_shots":   parsed["zone_shots"],
        "shot_ml_avg":  parsed["shot_ml_avg"],
        "zone_weights": zone_weights,
        "avg_delay":    parsed["avg_delay"],
        "source":       parsed["source"],
    }


# ─── Rally generator ──────────────────────────────────────────────────────────

PACE_FACTORS = {
    "warmup":   1.45,
    "moderate": 1.00,
    "fast":     0.62,
}

def sample_start(model):
    zones   = list(model["zone_weights"].keys())
    weights = list(model["zone_weights"].values())
    zone    = random.choices(zones, weights=weights, k=1)[0]
    shots   = model["zone_shots"].get(zone, {"drive": 1})
    shot    = random.choices(list(shots.keys()), weights=list(shots.values()), k=1)[0]
    return (zone, shot)


def sample_next(model, state):
    nexts = model["transitions"].get(state)
    if nexts:
        return random.choices(list(nexts.keys()),
                              weights=list(nexts.values()), k=1)[0]
    return sample_start(model)


def generate_rally(model, min_shots, max_shots, pace):
    pace_factor = PACE_FACTORS.get(pace, 1.0)
    length      = random.randint(min_shots, max_shots)
    state       = sample_start(model)
    rally       = []

    prev_zone, prev_shot = state

    for i in range(length):
        zone, shot = state

        if i == 0:
            # First shot: starting delay (already handled as inter-rally gap)
            delay = round(random.gauss(1.8, 0.2) * pace_factor, 2)
        else:
            ml_avg = model["shot_ml_avg"].get(shot, model["avg_delay"])
            delay  = compute_delay(prev_shot, prev_zone,
                                   shot, zone,
                                   pace_factor, ml_avg)

        rally.append({
            "zone":    zone,
            "shot":    shot,
            "delay_s": delay,
            "pace":    pace,
        })

        prev_zone, prev_shot = zone, shot
        state = sample_next(model, state)

    return rally


def generate_session(model, mode):
    if mode == "warmup":
        specs = [{"pace": "warmup", "min": 8, "max": 10}] * 5
        gap_range = (4.5, 6.5)
    else:
        specs = []
        for i in range(8):
            pace = "moderate" if i % 2 == 0 else "fast"
            specs.append({"pace": pace, "min": 8, "max": 15})
        gap_range = (3.0, 4.5)

    cues = []
    for i, spec in enumerate(specs):
        rally = generate_rally(model, spec["min"], spec["max"], spec["pace"])
        if rally:
            rally[0]["delay_s"]    = round(random.uniform(*gap_range), 2)
            rally[0]["rally_start"] = True
            rally[0]["rally_num"]   = i + 1
        cues.extend(rally)

    return cues, len(specs)


# ─── Flask app + HTML ─────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<title>AI Coach</title>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;900&family=Barlow:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#060a10;--surface:#0d1620;--border:#182535;
  --accent:#00d4ff;--green:#00e87a;--orange:#ff8c00;
  --red:#ff3434;--yellow:#ffd000;--text:#dce8f0;--sub:#445566;
  --font-d:'Barlow Condensed',sans-serif;--font-b:'Barlow',sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
html,body{height:100%;background:var(--bg);color:var(--text);
  font-family:var(--font-b);overflow:hidden}
#app{display:flex;flex-direction:column;height:100dvh;
  padding:10px 12px 14px;gap:9px}

/* Header */
header{display:flex;align-items:center;justify-content:space-between;flex-shrink:0}
.logo{font-family:var(--font-d);font-size:1.05rem;font-weight:900;
  letter-spacing:.2em;color:var(--accent);text-transform:uppercase}
.source-tag{font-size:.58rem;color:var(--sub);text-align:right;
  max-width:58%;line-height:1.4}

/* Mode bar */
#mode-bar{display:flex;gap:6px;flex-shrink:0}
.mode-btn{flex:1;padding:9px 4px;border-radius:8px;
  border:1.5px solid var(--border);background:var(--surface);
  color:var(--sub);font-family:var(--font-d);font-size:.85rem;
  font-weight:700;letter-spacing:.08em;cursor:pointer;
  transition:border-color .18s,color .18s;text-transform:uppercase;
  -webkit-appearance:none}
.mode-btn.active{border-color:var(--accent);color:var(--accent)}
.mode-btn:active{transform:scale(.97)}

/* Court */
#court-wrap{display:flex;justify-content:center;flex-shrink:0}
#court{display:grid;grid-template-columns:1fr 1fr;
  grid-template-rows:repeat(3,1fr);width:160px;height:205px;
  border-radius:8px;overflow:hidden;border:1.5px solid var(--border);
  gap:1px;background:var(--border);position:relative}
.zc{display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:2px;font-family:var(--font-d);
  font-size:1.8rem;font-weight:900;color:rgba(255,255,255,.13);
  background:var(--surface);transition:background .1s,color .1s;
  position:relative}
.zc .zl{font-size:.5rem;font-weight:600;letter-spacing:.06em;
  color:inherit;opacity:.5;text-transform:uppercase}
.zc.active{color:#000 !important}
.zc.z1.active{background:#00b4d8}.zc.z2.active{background:#0077b6}
.zc.z3.active{background:#00c853}.zc.z4.active{background:#009040}
.zc.z5.active{background:#ff8f00}.zc.z6.active{background:#e05000}
.zc.pop{animation:zpop .25s ease-out}
@keyframes zpop{0%{transform:scale(1.08)}100%{transform:scale(1)}}
#net-lbl{position:absolute;bottom:-16px;left:50%;
  transform:translateX(-50%);font-size:.5rem;letter-spacing:.14em;
  color:var(--sub);white-space:nowrap}

/* Cue box */
#cue-box{background:var(--surface);border:1px solid var(--border);
  border-radius:12px;padding:10px 14px 12px;text-align:center;
  flex-shrink:0;position:relative;overflow:hidden}
#cue-box::before{content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse at 50% 0%,
    rgba(0,212,255,.06),transparent 60%);pointer-events:none}
#rally-info{font-family:var(--font-d);font-size:.65rem;font-weight:700;
  letter-spacing:.12em;color:var(--sub);text-transform:uppercase}
#pace-chip{display:inline-block;margin:3px 0;font-family:var(--font-d);
  font-size:.62rem;font-weight:700;letter-spacing:.1em;padding:2px 10px;
  border-radius:20px;border:1px solid var(--border);color:var(--sub);
  text-transform:uppercase;transition:all .18s}
#pace-chip.warmup{border-color:#4a9eff;color:#4a9eff}
#pace-chip.moderate{border-color:var(--green);color:var(--green)}
#pace-chip.fast{border-color:var(--red);color:var(--red)}
#cue-num{font-family:var(--font-d);font-size:5.4rem;font-weight:900;
  line-height:.9;letter-spacing:-3px;color:var(--green);transition:color .12s}
#cue-num.back{color:var(--orange)}
#cue-num.mid{color:var(--yellow)}
#cue-shot-name{font-family:var(--font-d);font-size:1.6rem;font-weight:700;
  letter-spacing:.05em;text-transform:uppercase;color:var(--text);margin-top:3px}
#cue-timing{font-size:.65rem;color:var(--sub);margin-top:2px;letter-spacing:.04em}

/* Timer bar */
#timer-bg{height:3px;background:var(--border);border-radius:2px;
  overflow:hidden;flex-shrink:0}
#timer-bar{height:100%;border-radius:2px;background:var(--accent);
  width:0%;transition:width linear}

/* Progress */
#prog-row{display:flex;align-items:center;gap:6px;flex-shrink:0}
.pt{font-size:.62rem;color:var(--sub);white-space:nowrap}
#prog-bg{height:3px;background:var(--border);border-radius:2px;flex:1}
#prog-fill{height:100%;background:var(--sub);border-radius:2px;
  width:0%;transition:width .25s}

/* Buttons */
#btns{display:flex;gap:8px;flex-shrink:0}
.btn{flex:1;padding:15px 8px;border-radius:10px;border:none;cursor:pointer;
  font-family:var(--font-d);font-size:1.05rem;font-weight:900;
  letter-spacing:.06em;text-transform:uppercase;
  transition:opacity .15s,transform .1s;-webkit-appearance:none}
.btn:active{transform:scale(.96);opacity:.8}
.btn:disabled{opacity:.25;pointer-events:none}
#btn-go{background:var(--green);color:#001a0d;flex:2;font-size:1.15rem}
#btn-go.on{background:var(--orange);color:#1a0800}
#btn-skip,#btn-rst{background:var(--surface);color:var(--text);
  border:1px solid var(--border);max-width:52px;font-size:1.1rem}

/* Rest overlay */
#rest{display:none;position:fixed;inset:0;background:rgba(4,8,14,.93);
  flex-direction:column;align-items:center;justify-content:center;
  gap:8px;z-index:50}
#rest.show{display:flex}
#rest-lbl{font-family:var(--font-d);font-size:.85rem;font-weight:700;
  letter-spacing:.2em;color:var(--sub);text-transform:uppercase}
#rest-n{font-family:var(--font-d);font-size:5rem;font-weight:900;
  color:var(--accent);line-height:1}
#rest-next{font-size:.65rem;color:var(--sub);letter-spacing:.06em}

/* Finish */
#fin{display:none;position:fixed;inset:0;background:rgba(4,8,14,.97);
  flex-direction:column;align-items:center;justify-content:center;
  gap:14px;z-index:99;padding:24px}
#fin.show{display:flex}
#fin h2{font-family:var(--font-d);font-size:2.5rem;font-weight:900;
  color:var(--green);letter-spacing:.1em;text-transform:uppercase}
.fstat{font-size:.78rem;color:var(--sub);text-align:center;line-height:2.1}
.fstat strong{color:var(--text);font-weight:600}
#btn-again{padding:14px 40px;background:var(--accent);color:#001a22;
  border:none;border-radius:10px;font-family:var(--font-d);
  font-size:1.1rem;font-weight:900;letter-spacing:.08em;
  text-transform:uppercase;cursor:pointer}
</style>
</head>
<body>
<div id="app">
  <header>
    <div class="logo">AI Coach</div>
    <div class="source-tag" id="src">—</div>
  </header>

  <div id="mode-bar">
    <button class="mode-btn" id="btn-warmup"   onclick="setMode('warmup')">Warm-Up</button>
    <button class="mode-btn active" id="btn-practice" onclick="setMode('practice')">Practice</button>
  </div>

  <div id="court-wrap">
    <div id="court">
      <div class="zc z5" id="z5">5<span class="zl">Back L</span></div>
      <div class="zc z6" id="z6">6<span class="zl">Back R</span></div>
      <div class="zc z3" id="z3">3<span class="zl">Mid L</span></div>
      <div class="zc z4" id="z4">4<span class="zl">Mid R</span></div>
      <div class="zc z1" id="z1">1<span class="zl">Front L</span></div>
      <div class="zc z2" id="z2">2<span class="zl">Front R</span></div>
      <div id="net-lbl">▲ NET ▲</div>
    </div>
  </div>

  <div id="cue-box">
    <div id="rally-info">—</div>
    <div id="pace-chip">—</div>
    <div id="cue-num">—</div>
    <div id="cue-shot-name">Select mode &amp; press Start</div>
    <div id="cue-timing"></div>
  </div>

  <div id="timer-bg"><div id="timer-bar"></div></div>

  <div id="prog-row">
    <span class="pt" id="pl">0 / 0</span>
    <div id="prog-bg"><div id="prog-fill"></div></div>
    <span class="pt" id="pr">—</span>
  </div>

  <div id="btns">
    <button class="btn" id="btn-rst"  onclick="restart()">↺</button>
    <button class="btn" id="btn-go"   onclick="toggle()">Start</button>
    <button class="btn" id="btn-skip" onclick="skipRally()">⏭</button>
  </div>
</div>

<div id="rest">
  <div id="rest-lbl">Rest</div>
  <div id="rest-n">—</div>
  <div id="rest-next">Next rally incoming</div>
</div>

<div id="fin">
  <h2>Session Done</h2>
  <div class="fstat" id="fstat"></div>
  <button id="btn-again" onclick="restart()">Go Again</button>
</div>

<script>
const ZN={1:'Front Left',2:'Front Right',3:'Mid Left',
  4:'Mid Right',5:'Back Left',6:'Back Right'};

let cues=[],nRallies=0,idx=0,running=false;
let T=null,RI=null,mode='practice',shots=0,rallyN=0;

function setMode(m){
  if(running) pause();
  mode=m;
  document.querySelectorAll('.mode-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('btn-'+m).classList.add('active');
  reset();
}

function toggle(){ running?pause():start(); }

async function start(){
  if(!cues.length){
    const r=await fetch('/generate?mode='+mode);
    const d=await r.json();
    if(d.error){alert(d.error);return;}
    cues=d.cues; nRallies=d.rallies;
    document.getElementById('src').textContent=d.source;
    prog();
  }
  running=true;
  const b=document.getElementById('btn-go');
  b.textContent='Pause'; b.classList.add('on');
  next();
}

function pause(){
  running=false;
  clearTimeout(T); clearInterval(RI); clearBar();
  document.getElementById('rest').classList.remove('show');
  const b=document.getElementById('btn-go');
  b.textContent='Resume'; b.classList.remove('on');
}

function restart(){
  pause(); reset();
  document.getElementById('btn-go').textContent='Start';
  document.getElementById('fin').classList.remove('show');
}

function reset(){
  cues=[]; idx=0; shots=0; rallyN=0;
  hilite(null);
  document.getElementById('cue-num').textContent='—';
  document.getElementById('cue-num').className='';
  document.getElementById('cue-shot-name').textContent='Select mode & press Start';
  document.getElementById('cue-timing').textContent='';
  document.getElementById('rally-info').textContent='—';
  document.getElementById('pace-chip').textContent='—';
  document.getElementById('pace-chip').className='';
  prog(); clearBar();
}

function skipRally(){
  if(!cues.length) return;
  clearTimeout(T); clearInterval(RI); clearBar();
  document.getElementById('rest').classList.remove('show');
  let found=false;
  for(let i=idx+1;i<cues.length;i++){
    if(cues[i].rally_start){idx=i;found=true;break;}
  }
  if(!found){done();return;}
  if(running) next();
}

function next(){
  if(!running||idx>=cues.length){done();return;}
  const c=cues[idx];
  if(c.rally_start&&idx>0){
    showRest(c.delay_s,c.pace,c.rally_num);
  } else {
    const ms=c.delay_s*1000;
    animBar(ms);
    T=setTimeout(play,ms);
  }
}

function play(){
  if(!running||idx>=cues.length) return;
  const c=cues[idx];
  if(c.rally_start) rallyN=c.rally_num||(rallyN+1);

  hilite(c.zone);
  const ne=document.getElementById('cue-num');
  ne.textContent=c.zone;
  ne.className=c.zone>=5?'back':c.zone>=3?'mid':'';
  document.getElementById('cue-shot-name').textContent=c.shot.toUpperCase();
  document.getElementById('cue-timing').textContent=c.delay_s.toFixed(2)+'s';
  document.getElementById('rally-info').textContent=
    'Rally '+rallyN+' / '+nRallies;
  const chip=document.getElementById('pace-chip');
  chip.textContent=(c.pace||'').toUpperCase();
  chip.className=c.pace||'';

  shots++; idx++;
  prog();
  speak(c.zone+' '+c.shot);
  next();
}

function done(){
  running=false; clearBar();
  document.getElementById('btn-go').textContent='Start';
  document.getElementById('btn-go').classList.remove('on');
  hilite(null);
  document.getElementById('fstat').innerHTML=
    '<strong>'+nRallies+'</strong> rallies<br>'+
    '<strong>'+shots+'</strong> shots<br>'+
    'Mode: <strong>'+mode.toUpperCase()+'</strong>';
  document.getElementById('fin').classList.add('show');
}

function showRest(secs,pace,rn){
  clearInterval(RI);
  const o=document.getElementById('rest');
  const cn=document.getElementById('rest-n');
  document.getElementById('rest-next').textContent=
    'Rally '+rn+' — '+
    (pace==='fast'?'FAST':pace==='warmup'?'WARM-UP':'MODERATE')+' pace';
  o.classList.add('show');
  hilite(null); clearBar();
  let r=Math.round(secs);
  cn.textContent=r;
  RI=setInterval(()=>{
    r--;
    cn.textContent=Math.max(0,r);
    if(r<=0){
      clearInterval(RI);
      o.classList.remove('show');
      if(running) play();
    }
  },1000);
}

function animBar(ms){
  const b=document.getElementById('timer-bar');
  b.style.transition='none'; b.style.width='100%';
  requestAnimationFrame(()=>requestAnimationFrame(()=>{
    b.style.transition=`width ${ms}ms linear`; b.style.width='0%';
  }));
}
function clearBar(){
  const b=document.getElementById('timer-bar');
  b.style.transition='none'; b.style.width='0%';
}

function hilite(z){
  [1,2,3,4,5,6].forEach(n=>{
    const el=document.getElementById('z'+n);
    const was=el.classList.contains('active');
    el.classList.toggle('active',n===z);
    if(n===z&&!was){el.classList.remove('pop');void el.offsetWidth;el.classList.add('pop');}
  });
}

function prog(){
  const t=cues.length, pct=t?(idx/t*100):0;
  document.getElementById('prog-fill').style.width=pct+'%';
  document.getElementById('pl').textContent=idx+' / '+t;
  document.getElementById('pr').textContent=nRallies+' rallies';
}

function speak(txt){
  if(!('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();
  const u=new SpeechSynthesisUtterance(txt);
  u.rate=1.2; u.volume=1.0;
  window.speechSynthesis.speak(u);
}

fetch('/info').then(r=>r.json()).then(d=>{
  document.getElementById('src').textContent=d.source;
});
</script>
</body>
</html>"""


@app.route('/')
def index_route():
    return render_template_string(HTML)

@app.route('/info')
def info():
    return jsonify({"source": MODEL.get("source","—")})

@app.route('/generate')
def generate():
    try:
        mode = request.args.get('mode', 'practice')
        cues, n = generate_session(MODEL, mode)
        return jsonify({"cues": cues, "rallies": n,
                        "source": MODEL.get("source","—")})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─── Entry point ──────────────────────────────────────────────────────────────

def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close(); return ip
    except: return "localhost"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--port",   type=int, default=5000)
    args = parser.parse_args()

    parsed = load_report(args.report)
    global MODEL
    MODEL  = build_model(parsed)

    avg = MODEL["avg_delay"]
    ip  = get_ip()

    print(f"\n{'='*52}")
    print(f"  Badminton AI Coach")
    print(f"  Source      : {MODEL['source']}")
    print(f"\n  Timing engine (physics-aware):")
    print(f"    Back->front (not post-smash) : ~2.33s")
    print(f"    Post-smash net-shot           : ~1.30s")
    print(f"    Post-smash any front zone     : ~1.60s")
    print(f"    Drop/clear recovery bonus     : +0.50s")
    print(f"    Smash recovery bonus          : +0.00s  (fast reply)")
    print(f"\n  Pace scaling:")
    print(f"    Warm-up   : {avg*1.45:.2f}s base  (1.45x)")
    print(f"    Moderate  : {avg*1.00:.2f}s base  (1.00x)")
    print(f"    Fast      : {avg*0.62:.2f}s base  (0.62x)")
    print(f"\n  ML model:")
    print(f"    Transitions : {len(MODEL['transitions'])} states")
    print(f"    Zones       : {sorted(MODEL['zone_shots'].keys())}")
    print(f"    Shots       : {sorted(MODEL['shot_ml_avg'].keys())}")
    print(f"\n  Open on your phone:")
    print(f"  http://{ip}:{args.port}")
    print(f"  (Phone must be on same WiFi)")
    print(f"{'='*52}\n")

    app.run(host='0.0.0.0', port=args.port, debug=False)


if __name__ == '__main__':
    main()