// rallyEngine.js
// Physics-aware rally generator.
// Ported from ai_coach.py — identical timing logic.

// ─── Court geometry ───────────────────────────────────────────────────────────
const ZONE_ROW = { 1:0, 2:0, 3:1, 4:1, 5:2, 6:2 };
const ZONE_COL = { 1:0, 2:1, 3:0, 4:1, 5:0, 6:1 };

const SHOT_RECOVERY = {
  smash:     0.00,
  drop:      0.55,
  clear:     0.50,
  lift:      0.40,
  'net-shot': 0.10,
  'net-kill': 0.05,
  drive:     0.15,
};

const ZONE_NAMES = {
  1:'Front Left', 2:'Front Right',
  3:'Mid Left',   4:'Mid Right',
  5:'Back Left',  6:'Back Right',
};

const PACE_FACTORS = {
  warmup:   1.45,
  moderate: 1.00,
  fast:     0.62,
};

function travelTime(fz, tz) {
  return Math.abs(ZONE_ROW[fz] - ZONE_ROW[tz]) * 0.30
       + Math.abs(ZONE_COL[fz] - ZONE_COL[tz]) * 0.20;
}

function gauss(mean, std) {
  // Box-Muller
  const u = 1 - Math.random();
  const v = Math.random();
  return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// ─── Timing engine ────────────────────────────────────────────────────────────
function computeDelay(prevShot, prevZone, nextShot, nextZone, paceFactor, mlAvg) {
  let base;

  // Priority 1: post-smash -> net play (fastest combo)
  if (prevShot === 'smash' && (nextShot === 'net-shot' || nextShot === 'net-kill')) {
    base = clamp(gauss(1.30, 0.12), 0.90, 1.80);
    return +( base * paceFactor ).toFixed(2);
  }

  // Priority 2: post-smash -> any front zone
  if (prevShot === 'smash' && (nextZone === 1 || nextZone === 2)) {
    base = clamp(gauss(1.60, 0.18), 1.10, 2.00);
    return +( base * paceFactor ).toFixed(2);
  }

  // Priority 3: back -> front (not post-smash) avg 2.33s
  if ((prevZone === 5 || prevZone === 6) && (nextZone === 1 || nextZone === 2)) {
    base = clamp(gauss(2.33, 0.25), 1.60, 3.20);
    return +( base * paceFactor ).toFixed(2);
  }

  // Priority 4: front -> back (sprint back)
  if ((prevZone === 1 || prevZone === 2) && (nextZone === 5 || nextZone === 6)) {
    base = clamp(gauss(2.10, 0.22), 1.50, 3.00);
    return +( base * paceFactor ).toFixed(2);
  }

  // Default: blend ML avg + recovery + travel
  const recovery = SHOT_RECOVERY[prevShot] ?? 0.25;
  const travel   = travelTime(prevZone, nextZone);
  const ml       = mlAvg || 2.0;
  let delay      = ml * 0.6 + recovery * 0.8 + travel * 1.2 + gauss(0, ml * 0.12);

  if (nextShot === 'drop' || nextShot === 'clear') delay = Math.max(delay, 1.8);
  else if (nextShot === 'smash')                   delay = Math.max(delay, 1.4);
  else if (nextShot === 'net-shot' || nextShot === 'net-kill') delay = Math.min(delay, 2.2);

  delay = clamp(delay, 0.5, 6.0);
  return +( delay * paceFactor ).toFixed(2);
}

// ─── Markov sampling ──────────────────────────────────────────────────────────
function weightedChoice(items, weights) {
  const total = weights.reduce((a, b) => a + b, 0);
  let r = Math.random() * total;
  for (let i = 0; i < items.length; i++) {
    r -= weights[i];
    if (r <= 0) return items[i];
  }
  return items[items.length - 1];
}

function sampleStart(model) {
  const zones   = Object.keys(model.zoneWeights).map(Number);
  const weights = zones.map(z => model.zoneWeights[z]);
  const zone    = weightedChoice(zones, weights);
  const shots   = model.zoneShots[zone];
  const sKeys   = Object.keys(shots);
  const sWts    = sKeys.map(s => shots[s]);
  const shot    = weightedChoice(sKeys, sWts);
  return { zone, shot };
}

function sampleNext(model, zone, shot) {
  const key   = `${zone}-${shot}`;
  const nexts = model.transitions[key];
  if (!nexts || Object.keys(nexts).length === 0) return sampleStart(model);
  const keys  = Object.keys(nexts);
  const wts   = keys.map(k => nexts[k]);
  const next  = weightedChoice(keys, wts);
  const [nz, ...rest] = next.split('-');
  return { zone: parseInt(nz), shot: rest.join('-') };
}

// ─── Rally generator ──────────────────────────────────────────────────────────
function generateRally(model, minShots, maxShots, pace) {
  const pf     = PACE_FACTORS[pace] ?? 1.0;
  const length = minShots + Math.floor(Math.random() * (maxShots - minShots + 1));
  let   cur    = sampleStart(model);
  const rally  = [];

  for (let i = 0; i < length; i++) {
    const { zone, shot } = cur;
    let delay;

    if (i === 0) {
      delay = +( clamp(gauss(1.8, 0.2), 1.2, 2.8) * pf ).toFixed(2);
    } else {
      const prev  = rally[rally.length - 1];
      const mlAvg = model.shotAvg[shot] ?? model.avgDelay;
      delay = computeDelay(prev.shot, prev.zone, shot, zone, pf, mlAvg);
    }

    rally.push({ zone, shot, delay_s: delay, pace,
                 zoneName: ZONE_NAMES[zone] });
    cur = sampleNext(model, zone, shot);
  }
  return rally;
}

export function generateSession(model, mode) {
  let specs, gapRange;
  if (mode === 'warmup') {
    specs    = Array(5).fill({ pace:'warmup', min:8, max:10 });
    gapRange = [4.5, 6.5];
  } else {
    specs = Array.from({ length:8 }, (_, i) => ({
      pace: i % 2 === 0 ? 'moderate' : 'fast',
      min: i % 2 === 0 ? 8 : 10,
      max: 15,
    }));
    gapRange = [3.0, 4.5];
  }

  const cues = [];
  specs.forEach((spec, i) => {
    const rally = generateRally(model, spec.min, spec.max, spec.pace);
    if (rally.length > 0) {
      const gap = gapRange[0] + Math.random() * (gapRange[1] - gapRange[0]);
      rally[0].delay_s    = +gap.toFixed(2);
      rally[0].rallyStart = true;
      rally[0].rallyNum   = i + 1;
    }
    cues.push(...rally);
  });

  return { cues, nRallies: specs.length };
}

// ─── Report parser ────────────────────────────────────────────────────────────
export function parseReport(data) {
  // Accepts the structured JSON format
  const model = {
    transitions: {},
    zoneShots:   {},
    zoneWeights: {},
    shotAvg:     {},
    avgDelay:    data.timing_overview?.average_delay_seconds ?? 2.41,
    source:      data.source ?? 'Unknown',
  };

  // Shot averages
  (data.shot_frequency ?? []).forEach(s => {
    model.shotAvg[s.shot] = s.avg_seconds;
  });

  // Zone -> shot weights
  (data.shots_by_zone ?? []).forEach(z => {
    const shots = {};
    (z.shots ?? []).forEach(s => { shots[s.shot] = s.count; });
    model.zoneShots[z.zone] = shots;
    model.zoneWeights[z.zone] = Object.values(shots).reduce((a,b)=>a+b,0);
  });

  // Transitions: normalise by from-state
  const raw = {};
  (data.top_shot_transitions ?? []).forEach(t => {
    const from = t.from;
    if (!raw[from]) raw[from] = {};
    raw[from][t.to] = t.probability_percentage / 100;
  });
  Object.keys(raw).forEach(from => {
    const total = Object.values(raw[from]).reduce((a,b)=>a+b,0);
    model.transitions[from] = {};
    Object.keys(raw[from]).forEach(to => {
      model.transitions[from][to] = raw[from][to] / total;
    });
  });

  return model;
}

export { ZONE_NAMES, PACE_FACTORS };
