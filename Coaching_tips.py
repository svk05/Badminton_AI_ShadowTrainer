"""
Badminton AI Coach — Player vs Pro Comparison
===============================================
Compares a player's cue JSON against a pro's structured report JSON.
Outputs gap analysis and prioritised coaching tips.

Usage:
    python compare_player.py --player me1_cues.json --pro syq_report.json
    python compare_player.py --player me1_cues.json --pro syq_report.json --output comparison.txt
"""

import json
import math
import argparse
from collections import defaultdict, Counter


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_player(path):
    """Load a raw cues JSON (from batch_analyse / manual labelling)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_pro(path):
    """Load a structured report JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─── Build player stats from raw cues ────────────────────────────────────────

def build_player_stats(data):
    # With this:
    cues   = data["cues"] if isinstance(data["cues"], list) else data.get("cues_list", [])
    if not cues:
        # Try alternative key names
        for key in ("shots", "data", "entries"):
            if isinstance(data.get(key), list):
                cues = data[key]
                break
    n      = len(cues)
    delays = [c["delay_s"] for c in cues]

    shot_counts  = Counter(c["shot"]  for c in cues)
    zone_counts  = Counter(c["zone"]  for c in cues)
    shot_delays  = defaultdict(list)
    zone_delays  = defaultdict(list)
    for c in cues:
        shot_delays[c["shot"]].append(c["delay_s"])
        zone_delays[c["zone"]].append(c["delay_s"])

    zone_shots = defaultdict(Counter)
    for c in cues:
        zone_shots[c["zone"]][c["shot"]] += 1

    # Transitions
    transitions = defaultdict(Counter)
    for i in range(len(cues) - 1):
        fk = f"{cues[i]['zone']}-{cues[i]['shot']}"
        tk = f"{cues[i+1]['zone']}-{cues[i+1]['shot']}"
        transitions[fk][tk] += 1

    # Sequences
    states   = [(c["zone"], c["shot"]) for c in cues]
    bigrams  = Counter(tuple(states[i:i+2]) for i in range(len(states)-1))
    trigrams = Counter(tuple(states[i:i+3]) for i in range(len(states)-2))

    front = sum(zone_counts.get(z,0) for z in (1,2)) / n * 100
    mid   = sum(zone_counts.get(z,0) for z in (3,4)) / n * 100
    back  = sum(zone_counts.get(z,0) for z in (5,6)) / n * 100

    return {
        "source":       data.get("source","player"),
        "n":            n,
        "avg_delay":    sum(delays)/n,
        "min_delay":    min(delays),
        "max_delay":    max(delays),
        "shot_counts":  shot_counts,
        "zone_counts":  zone_counts,
        "shot_delays":  {s: sum(d)/len(d) for s,d in shot_delays.items()},
        "zone_delays":  {z: sum(d)/len(d) for z,d in zone_delays.items()},
        "zone_shots":   zone_shots,
        "transitions":  transitions,
        "bigrams":      bigrams,
        "trigrams":     trigrams,
        "coverage":     {"front":front,"mid":mid,"back":back},
    }


# ─── Build pro stats from structured report ───────────────────────────────────

def build_pro_stats(data):
    t   = data["timing_overview"]
    n   = data["cues"]

    shot_counts = {s["shot"]: s["count"]       for s in data["shot_frequency"]}
    shot_delays = {s["shot"]: s["avg_seconds"]  for s in data["shot_frequency"]}
    zone_counts = {z["zone"]: z["count"]        for z in data["zone_usage"]}
    zone_delays = {z["zone"]: z["avg_seconds"]  for z in data["zone_usage"]}

    zone_shots = {}
    for z in data["shots_by_zone"]:
        zone_shots[z["zone"]] = Counter({s["shot"]: s["count"] for s in z["shots"]})

    transitions = defaultdict(Counter)
    for tr in data["top_shot_transitions"]:
        transitions[tr["from"]][tr["to"]] = tr["probability_percentage"]

    cc = data.get("court_coverage",{})

    return {
        "source":       data.get("source","pro"),
        "n":            n,
        "avg_delay":    t["average_delay_seconds"],
        "min_delay":    t["fastest_exchange_seconds"],
        "max_delay":    t["slowest_exchange_seconds"],
        "shot_counts":  shot_counts,
        "zone_counts":  zone_counts,
        "shot_delays":  shot_delays,
        "zone_delays":  zone_delays,
        "zone_shots":   zone_shots,
        "transitions":  transitions,
        "bigrams":      Counter({tuple(s["sequence"][i].replace("Zone","").split() for i in range(2))
                                 if False else tuple(s["sequence"]): s["count"]
                                 for s in data.get("most_common_2_shot_sequences",[])}),
        "coverage":     {
            "front": cc.get("front_percentage",0),
            "mid":   cc.get("mid_percentage",0),
            "back":  cc.get("back_percentage",0),
        },
        "coaching_notes":    data.get("coaching_notes",[]),
        "key_pattern":       data.get("key_pattern_to_drill",""),
    }


# ─── Comparison engine ────────────────────────────────────────────────────────

def pct_share(counts, key):
    total = sum(counts.values())
    return counts.get(key, 0) / total * 100 if total else 0

def delta_str(player_val, pro_val, unit=""):
    diff = player_val - pro_val
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.1f}{unit}"

def compare(player, pro):
    tips   = []   # list of (priority, category, message)
    lines  = []   # report lines

    def h1(t): lines.append("\n" + "="*60 + f"\n  {t}\n" + "="*60)
    def h2(t): lines.append(f"\n  {'─'*50}\n  {t}\n  {'─'*50}")
    def row(t): lines.append("  " + t)
    def blank(): lines.append("")

    all_shots = sorted(set(list(player["shot_counts"].keys()) +
                           list(pro["shot_counts"].keys())))
    all_zones = sorted(set(list(player["zone_counts"].keys()) +
                           list(pro["zone_counts"].keys())))

    # ── Header ────────────────────────────────────────────────────────────────
    h1("PLAYER vs PRO — GAP ANALYSIS")
    row(f"Player : {player['source']}  ({player['n']} cues)")
    row(f"Pro    : {pro['source']}  ({pro['n']} cues)")

    # ── Pace comparison ───────────────────────────────────────────────────────
    h2("PACE")
    row(f"{'':20} {'Player':>10}  {'Pro':>10}  {'Gap':>10}")
    row(f"{'─'*52}")

    p_avg, r_avg = player["avg_delay"], pro["avg_delay"]
    row(f"{'Avg delay':<20} {p_avg:>9.2f}s  {r_avg:>9.2f}s  {delta_str(p_avg,r_avg,'s'):>10}")

    pace_diff = p_avg - r_avg
    if pace_diff > 0.6:
        tips.append((1, "Pace",
            f"Your avg pace ({p_avg:.2f}s) is {pace_diff:.2f}s SLOWER than the pro ({r_avg:.2f}s). "
            f"Focus on faster recovery to centre court after each shot."))
    elif pace_diff < -0.3:
        tips.append((3, "Pace",
            f"Your pace ({p_avg:.2f}s) is faster than the pro. "
            f"Ensure you're not rushing — controlled shot preparation is key."))
    else:
        row(f"  ✓ Pace is close to pro level.")

    # ── Shot frequency comparison ─────────────────────────────────────────────
    h2("SHOT FREQUENCY  (% of total shots)")
    row(f"{'Shot':<12} {'Player':>8}  {'Pro':>8}  {'Gap':>8}  {'Verdict'}")
    row(f"{'─'*56}")

    shot_gaps = []
    for shot in all_shots:
        p_pct = pct_share(player["shot_counts"], shot)
        r_pct = pct_share(pro["shot_counts"],    shot)
        diff  = p_pct - r_pct
        verdict = ""
        if abs(diff) > 8:
            verdict = "▲ too much" if diff > 0 else "▼ too little"
            shot_gaps.append((abs(diff), shot, diff, p_pct, r_pct))
        row(f"{shot:<12} {p_pct:>7.1f}%  {r_pct:>7.1f}%  {delta_str(p_pct,r_pct,'%'):>8}  {verdict}")

    blank()
    for _, shot, diff, p_pct, r_pct in sorted(shot_gaps, reverse=True)[:4]:
        if diff > 0:
            tips.append((2, "Shot mix",
                f"You play {shot} {p_pct:.0f}% of the time vs pro's {r_pct:.0f}%. "
                f"You're over-relying on {shot} — diversify your shot selection."))
        else:
            tips.append((2, "Shot mix",
                f"You play {shot} only {p_pct:.0f}% vs pro's {r_pct:.0f}%. "
                f"Incorporate more {shot}s into your game."))

    # ── Shot timing comparison ────────────────────────────────────────────────
    h2("SHOT TIMING  (avg seconds per shot type)")
    row(f"{'Shot':<12} {'Player':>8}  {'Pro':>8}  {'Gap':>8}")
    row(f"{'─'*42}")

    for shot in all_shots:
        p_t = player["shot_delays"].get(shot)
        r_t = pro["shot_delays"].get(shot)
        if p_t is None or r_t is None: continue
        diff = p_t - r_t
        flag = "  ← SLOW" if diff > 0.5 else ("  ← FAST" if diff < -0.4 else "")
        row(f"{shot:<12} {p_t:>7.2f}s  {r_t:>7.2f}s  {delta_str(p_t,r_t,'s'):>8}{flag}")
        if diff > 0.5:
            tips.append((2, "Shot timing",
                f"Your {shot} takes {p_t:.2f}s avg vs pro's {r_t:.2f}s. "
                f"Work on faster preparation and swing for {shot}."))

    # ── Court coverage comparison ─────────────────────────────────────────────
    h2("COURT COVERAGE")
    row(f"{'Area':<12} {'Player':>8}  {'Pro':>8}  {'Gap':>8}")
    row(f"{'─'*42}")

    for area in ("front","mid","back"):
        p_c = player["coverage"][area]
        r_c = pro["coverage"][area]
        row(f"{area.capitalize():<12} {p_c:>7.1f}%  {r_c:>7.1f}%  {delta_str(p_c,r_c,'%'):>8}")
        diff = p_c - r_c
        if area == "back" and diff < -8:
            tips.append((2, "Court coverage",
                f"You spend only {p_c:.0f}% of time in back court vs pro's {r_c:.0f}%. "
                f"Practice back-court footwork and overhead shots."))
        if area == "front" and diff < -8:
            tips.append((2, "Court coverage",
                f"You spend only {p_c:.0f}% in front court vs pro's {r_c:.0f}%. "
                f"Work on net play and forward movement."))
        if area == "mid" and diff > 10:
            tips.append((3, "Court coverage",
                f"You spend {p_c:.0f}% in mid-court vs pro's {r_c:.0f}%. "
                f"You may be hesitating — commit fully to front or back."))

    # ── Zone usage comparison ─────────────────────────────────────────────────
    h2("ZONE USAGE  (% of shots per zone)")
    row(f"{'Zone':<20} {'Player':>8}  {'Pro':>8}  {'Gap':>8}")
    row(f"{'─'*48}")

    ZONE_NAMES = {1:"Front-Left",2:"Front-Right",3:"Mid-Left",
                  4:"Mid-Right",5:"Back-Left",6:"Back-Right"}
    for z in all_zones:
        p_z = pct_share(player["zone_counts"], z)
        r_z = pct_share(pro["zone_counts"],    z)
        name = ZONE_NAMES.get(z,str(z))
        diff = p_z - r_z
        flag = "  ▲" if diff > 8 else ("  ▼" if diff < -8 else "")
        row(f"Zone {z} ({name:<12}) {p_z:>7.1f}%  {r_z:>7.1f}%  {delta_str(p_z,r_z,'%'):>8}{flag}")

    # ── Shots by zone comparison ───────────────────────────────────────────────
    h2("SHOT SELECTION BY ZONE  (vs pro)")
    for z in all_zones:
        p_zone = player["zone_shots"].get(z, Counter())
        r_zone = pro["zone_shots"].get(z, Counter())
        if not p_zone and not r_zone: continue

        p_total = sum(p_zone.values()) or 1
        r_total = sum(r_zone.values()) or 1
        name    = ZONE_NAMES.get(z,str(z))
        row(f"Zone {z} ({name}):")

        all_zone_shots = sorted(set(list(p_zone.keys())+list(r_zone.keys())))
        for shot in all_zone_shots:
            p_s = p_zone.get(shot,0)/p_total*100
            r_s = r_zone.get(shot,0)/r_total*100
            diff = p_s - r_s
            flag = "  ← over" if diff>15 else ("  ← under" if diff<-15 else "")
            row(f"    {shot:<12} {p_s:>5.0f}%  vs  {r_s:>5.0f}% pro{flag}")
            if diff < -15 and r_s > 15:
                tips.append((3, f"Zone {z} shot mix",
                    f"In Zone {z} ({name}) you barely play {shot} ({p_s:.0f}%) "
                    f"but the pro uses it {r_s:.0f}% there. Add {shot} from Zone {z}."))
            if diff > 20:
                tips.append((3, f"Zone {z} shot mix",
                    f"In Zone {z} ({name}) you over-use {shot} ({p_s:.0f}%) "
                    f"vs pro's {r_s:.0f}%. Mix in other options."))

    # ── Transition comparison ─────────────────────────────────────────────────
    h2("TOP TRANSITIONS YOU USE  (vs pro)")
    p_trans_all = []
    for fk, nexts in player["transitions"].items():
        total = sum(nexts.values())
        for tk, cnt in nexts.most_common(2):
            p_trans_all.append((fk, tk, cnt, cnt/total*100))
    p_trans_all.sort(key=lambda x: -x[2])

    row(f"{'From':<18} {'To':<18} {'Your%':>7}  {'Pro%':>7}")
    row(f"{'─'*54}")
    for fk, tk, cnt, p_prob in p_trans_all[:10]:
        r_prob = pro["transitions"].get(fk,{}).get(tk, 0)
        row(f"{fk:<18} {tk:<18} {p_prob:>6.0f}%  {r_prob:>6.0f}%")

    # ── Key patterns you're missing ────────────────────────────────────────────
    h2("KEY PRO PATTERNS — ARE YOU DRILLING THEM?")
    if pro.get("key_pattern"):
        row(f"Pro's key drill : {pro['key_pattern']}")
        # Check if player uses any part of this pattern
        parts = [p.strip() for p in pro["key_pattern"].split("->")]
        row(f"Pattern shots   : {' -> '.join(parts)}")
        tips.append((1, "Key pattern",
            f"The pro's most repeated pattern is: {pro['key_pattern']}. "
            f"Specifically drill this sequence in shadow practice."))
    blank()

    # ── Strengths ─────────────────────────────────────────────────────────────
    h2("YOUR STRENGTHS  (where you match or exceed the pro)")
    strengths_found = False
    for shot in all_shots:
        p_t = player["shot_delays"].get(shot)
        r_t = pro["shot_delays"].get(shot)
        if p_t and r_t and abs(p_t - r_t) <= 0.3:
            row(f"  ✓ {shot} timing ({p_t:.2f}s) is close to pro level ({r_t:.2f}s)")
            strengths_found = True

    p_cov = player["coverage"]
    r_cov = pro["coverage"]
    for area in ("front","mid","back"):
        if abs(p_cov[area] - r_cov[area]) <= 5:
            row(f"  ✓ {area.capitalize()} court coverage ({p_cov[area]:.0f}%) matches pro ({r_cov[area]:.0f}%)")
            strengths_found = True
    if not strengths_found:
        row("  Keep training — more data will reveal your strengths.")

    # ── Prioritised coaching tips ──────────────────────────────────────────────
    h1("PRIORITISED COACHING TIPS")
    tips.sort(key=lambda x: x[0])
    for i, (priority, category, msg) in enumerate(tips, 1):
        p_str = "🔴 HIGH" if priority==1 else ("🟡 MED" if priority==2 else "🟢 LOW")
        row(f"[{i:02d}] {p_str}  [{category}]")
        # Word wrap at ~60 chars
        words = msg.split()
        line  = "       "
        for w in words:
            if len(line)+len(w) > 68:
                row(line); line = "       " + w + " "
            else:
                line += w + " "
        if line.strip(): row(line)
        blank()

    lines.append("="*60 + "\n")
    return "\n".join(lines), tips


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", required=True,
                        help="Your cues JSON (raw format from batch_analyse)")
    parser.add_argument("--pro",    required=True,
                        help="Pro structured report JSON")
    parser.add_argument("--output", default=None,
                        help="Save report to this .txt file")
    args = parser.parse_args()

    print(f"\nLoading player data : {args.player}")
    player_raw  = load_player(args.player)
    player_stats = build_player_stats(player_raw)

    print(f"Loading pro data    : {args.pro}")
    pro_raw  = load_pro(args.pro)
    pro_stats = build_pro_stats(pro_raw)

    print(f"\nComparing {player_stats['n']} player cues vs {pro_stats['n']} pro cues...\n")

    report, tips = compare(player_stats, pro_stats)
    print(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved → {args.output}\n")

    print(f"Summary: {len(tips)} coaching tips generated "
          f"({sum(1 for t in tips if t[0]==1)} high priority, "
          f"{sum(1 for t in tips if t[0]==2)} medium, "
          f"{sum(1 for t in tips if t[0]==3)} low)\n")


if __name__ == "__main__":
    main()