"""
Badminton Shadow Coach — Rally Pattern Analyser
=================================================
Analyses cues.json and outputs a human-readable trainer report.

Usage:
    python rally_ml.py --input cues.json
    python rally_ml.py --input cues.json --output trainer_report.txt
    python rally_ml.py --input cues.json --output trainer_report.txt --name "Shi Yu Qi"
"""

import json
import math
import argparse
from collections import defaultdict, Counter


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_state(c):
    return (c["zone"], c["shot"])

def std(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean)**2 for v in values) / len(values))

def entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((c/total) * math.log2(c/total) for c in counts if c > 0)

ZONE_NAMES = {
    1:"Front-Left", 2:"Front-Right",
    3:"Mid-Left",   4:"Mid-Right",
    5:"Back-Left",  6:"Back-Right",
}


# ─── Analysis ─────────────────────────────────────────────────────────────────

def analyse(cues):
    states  = [make_state(c) for c in cues]
    shots   = [c["shot"]     for c in cues]
    zones   = [c["zone"]     for c in cues]
    delays  = [c["delay_s"]  for c in cues]

    # Transitions
    transitions = defaultdict(Counter)
    timings     = defaultdict(list)
    for i in range(len(cues) - 1):
        s, sn = states[i], states[i+1]
        transitions[s][sn] += 1
        timings[s].append(cues[i]["delay_s"])
    timings[states[-1]].append(cues[-1]["delay_s"])

    # N-grams
    bigrams  = Counter(tuple(states[i:i+2]) for i in range(len(states)-1))
    trigrams = Counter(tuple(states[i:i+3]) for i in range(len(states)-2))

    # Shot-level timing
    shot_delays = defaultdict(list)
    zone_delays = defaultdict(list)
    for c in cues:
        shot_delays[c["shot"]].append(c["delay_s"])
        zone_delays[c["zone"]].append(c["delay_s"])

    # Shot -> zone distribution
    shot_zones = defaultdict(Counter)
    for c in cues:
        shot_zones[c["shot"]][c["zone"]] += 1

    # Zone -> shot distribution
    zone_shots = defaultdict(Counter)
    for c in cues:
        zone_shots[c["zone"]][c["shot"]] += 1

    # Entropy per state (predictability)
    state_entropy = {}
    for state, nexts in transitions.items():
        state_entropy[state] = entropy(list(nexts.values()))

    return {
        "states": states, "shots": shots, "zones": zones, "delays": delays,
        "transitions": transitions, "timings": timings,
        "bigrams": bigrams, "trigrams": trigrams,
        "shot_delays": shot_delays, "zone_delays": zone_delays,
        "shot_zones": shot_zones, "zone_shots": zone_shots,
        "state_entropy": state_entropy,
    }


# ─── Report builder ───────────────────────────────────────────────────────────

def build_report(data, cues, source, player_name):
    A      = analyse(cues)
    lines  = []
    n      = len(cues)
    delays = A["delays"]

    def h1(text): lines.append("\n" + "="*60 + f"\n  {text}\n" + "="*60)
    def h2(text): lines.append(f"\n  {'─'*50}\n  {text}\n  {'─'*50}")
    def row(text): lines.append("  " + text)
    def blank(): lines.append("")

    # ── Title ─────────────────────────────────────────────────────────────────
    h1(f"BADMINTON TRAINER REPORT")
    row(f"Player  : {player_name}")
    row(f"Source  : {source}")
    row(f"Cues    : {n}")
    row(f"States  : {len(set(A['states']))}  unique (zone + shot combinations)")

    # ── Timing overview ───────────────────────────────────────────────────────
    h2("TIMING OVERVIEW")
    row(f"Average delay between shots : {sum(delays)/n:.2f}s")
    row(f"Fastest exchange            : {min(delays):.2f}s")
    row(f"Slowest exchange            : {max(delays):.2f}s")
    row(f"Std deviation               : {std(delays):.2f}s")
    blank()
    row("Interpretation:")
    avg = sum(delays)/n
    if avg < 1.5:
        row("  Very fast-paced style. Minimal recovery time between shots.")
    elif avg < 2.5:
        row("  Moderate pace. Balanced between attack and recovery.")
    else:
        row("  Deliberate pace. Player takes time to reset and reposition.")

    # ── Shot distribution ─────────────────────────────────────────────────────
    h2("SHOT FREQUENCY")
    shot_counts = Counter(A["shots"])
    for shot, count in sorted(shot_counts.items(), key=lambda x: -x[1]):
        pct = count / n * 100
        bar = "█" * count
        sd  = A["shot_delays"][shot]
        row(f"{shot:<10} {count:>3}x  {pct:5.1f}%  avg {sum(sd)/len(sd):.2f}s  {bar}")
    blank()
    top_shot = shot_counts.most_common(1)[0][0]
    row(f"Dominant shot: {top_shot}  — forms the core of this player's game.")

    # ── Zone distribution ─────────────────────────────────────────────────────
    h2("ZONE USAGE")
    zone_counts = Counter(A["zones"])
    for zone in sorted(zone_counts.keys()):
        count = zone_counts[zone]
        pct   = count / n * 100
        bar   = "█" * count
        zd    = A["zone_delays"][zone]
        row(f"Zone {zone} ({ZONE_NAMES[zone]:<12}) {count:>3}x  {pct:5.1f}%  "
            f"avg {sum(zd)/len(zd):.2f}s  {bar}")
    blank()
    top_zone = zone_counts.most_common(1)[0][0]
    row(f"Most active zone: Zone {top_zone} ({ZONE_NAMES[top_zone]})")
    front = sum(zone_counts.get(z,0) for z in (1,2))
    mid   = sum(zone_counts.get(z,0) for z in (3,4))
    back  = sum(zone_counts.get(z,0) for z in (5,6))
    row(f"Court coverage  : Front {front/n*100:.0f}%  Mid {mid/n*100:.0f}%  Back {back/n*100:.0f}%")

    # ── Shot from zone breakdown ───────────────────────────────────────────────
    h2("SHOTS BY ZONE")
    for zone in sorted(A["zone_shots"].keys()):
        shots_here = A["zone_shots"][zone]
        total_here = sum(shots_here.values())
        parts = ", ".join(
            f"{s} {c}x ({c/total_here*100:.0f}%)"
            for s, c in shots_here.most_common()
        )
        row(f"Zone {zone} ({ZONE_NAMES[zone]:<12}): {parts}")

    # ── Top transitions ────────────────────────────────────────────────────────
    h2("TOP SHOT TRANSITIONS  (what follows what)")
    all_trans = []
    for state, nexts in A["transitions"].items():
        for next_state, count in nexts.items():
            total = sum(nexts.values())
            all_trans.append((state, next_state, count, count/total*100))
    all_trans.sort(key=lambda x: -x[2])
    row(f"{'From':<18} {'To':<18} {'Count':>5}  {'Prob':>6}")
    row(f"{'─'*18} {'─'*18} {'─'*5}  {'─'*6}")
    for s, ns, count, prob in all_trans[:15]:
        s_str  = f"{s[0]}-{s[1]}"
        ns_str = f"{ns[0]}-{ns[1]}"
        row(f"{s_str:<18} {ns_str:<18} {count:>5}x  {prob:>5.0f}%")

    # ── Top bigrams ────────────────────────────────────────────────────────────
    h2("MOST COMMON 2-SHOT SEQUENCES")
    for seq, count in A["bigrams"].most_common(10):
        seq_str = "  ->  ".join(f"Zone{z} {s}" for z,s in seq)
        row(f"  {seq_str}  ({count}x)")

    # ── Top trigrams ───────────────────────────────────────────────────────────
    h2("MOST COMMON 3-SHOT SEQUENCES")
    for seq, count in A["trigrams"].most_common(10):
        seq_str = "  ->  ".join(f"Zone{z} {s}" for z,s in seq)
        row(f"  {seq_str}  ({count}x)")

    # ── Predictability ─────────────────────────────────────────────────────────
    h2("STATE PREDICTABILITY  (entropy: 0=always same next shot, high=varied)")
    sorted_ent = sorted(A["state_entropy"].items(), key=lambda x: x[1])
    row("  Most predictable states (low entropy):")
    for state, ent in sorted_ent[:5]:
        nexts = A["transitions"][state]
        top_next = nexts.most_common(1)[0]
        total    = sum(nexts.values())
        row(f"    Zone{state[0]} {state[1]:<10}  entropy={ent:.2f}  "
            f"-> usually Zone{top_next[0][0]} {top_next[0][1]} "
            f"({top_next[1]/total*100:.0f}%)")
    blank()
    row("  Most varied states (high entropy):")
    for state, ent in sorted_ent[-5:]:
        row(f"    Zone{state[0]} {state[1]:<10}  entropy={ent:.2f}  "
            f"(many different follow-ups)")

    # ── Coaching notes ─────────────────────────────────────────────────────────
    h2("COACHING NOTES")

    # Smash rate
    smash_pct = shot_counts.get("smash",0) / n * 100
    if smash_pct > 20:
        row(f"  HIGH smash rate ({smash_pct:.0f}%). Aggressive attacking style.")
        row(f"  Shadow focus: back-court positioning and overhead prep.")
    elif smash_pct > 10:
        row(f"  Moderate smash rate ({smash_pct:.0f}%). Mixed attack/rally.")
    else:
        row(f"  Low smash rate ({smash_pct:.0f}%). More net-play / rally focused.")

    # Net dominance
    net_shots = shot_counts.get("net-shot",0) + shot_counts.get("net-kill",0)
    net_pct   = net_shots / n * 100
    if net_pct > 25:
        row(f"  Strong net presence ({net_pct:.0f}% net shots/kills).")
        row(f"  Shadow focus: quick transitions to Zone 1/2.")

    # Lift tendency
    lift_pct = shot_counts.get("lift",0) / n * 100
    if lift_pct > 20:
        row(f"  Frequent lifts ({lift_pct:.0f}%). Resets rally often.")
        row(f"  Shadow focus: defensive positioning and recovery speed.")

    # Pace note
    if avg < 1.8:
        row(f"  Very quick rally pace (avg {avg:.2f}s). Build fitness for fast exchanges.")
    elif avg > 2.5:
        row(f"  Deliberate pace (avg {avg:.2f}s). Focus on court coverage and control.")

    blank()
    row("  Key pattern to drill:")
    if A["trigrams"]:
        top_trig = A["trigrams"].most_common(1)[0][0]
        trig_str = " -> ".join(f"Zone{z} {s}" for z,s in top_trig)
        row(f"  {trig_str}")
        row(f"  This is the most repeated 3-shot sequence in this footage.")

    blank()
    lines.append("="*60)
    lines.append("")

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="cues.json")
    parser.add_argument("--output", default=None,  help="trainer_report.txt (optional)")
    parser.add_argument("--name",   default="Unknown", help="Player name for report")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    cues   = data["cues"]
    source = data.get("source", args.input)

    report = build_report(data, cues, source, args.name)

    # Always print to terminal
    print(report)

    # Optionally save
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  Report saved -> {args.output}\n")


if __name__ == "__main__":
    main()