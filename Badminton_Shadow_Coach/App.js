// App.js
import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity, Animated,
  StatusBar, Alert, ScrollView, ActivityIndicator,
} from 'react-native';
import * as DocumentPicker from 'expo-document-picker';
import * as FileSystem from 'expo-file-system';
import * as Speech from 'expo-speech';
import * as KeepAwake from 'expo-keep-awake';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { generateSession, parseReport, ZONE_NAMES } from './src/rallyEngine';

// ─── Theme ────────────────────────────────────────────────────────────────────
const C = {
  bg:      '#060a10',
  surface: '#0d1620',
  border:  '#182535',
  accent:  '#00d4ff',
  green:   '#00e87a',
  orange:  '#ff8c00',
  red:     '#ff3434',
  yellow:  '#ffd000',
  text:    '#dce8f0',
  sub:     '#445566',
};

const ZONE_COLORS = {
  1: '#00b4d8', 2: '#0077b6',
  3: '#00c853', 4: '#009040',
  5: '#ff8f00', 6: '#e05000',
};

const PACE_COLORS = {
  warmup:   '#4a9eff',
  moderate: C.green,
  fast:     C.red,
};

// ─── Court diagram ────────────────────────────────────────────────────────────
function CourtDiagram({ activeZone }) {
  // Grid order: back-left, back-right, mid-left, mid-right, front-left, front-right
  const zones = [5, 6, 3, 4, 1, 2];
  return (
    <View style={s.courtWrap}>
      <View style={s.court}>
        {zones.map(z => {
          const active = z === activeZone;
          return (
            <View key={z} style={[
              s.zoneCell,
              active && { backgroundColor: ZONE_COLORS[z] },
            ]}>
              <Text style={[s.zoneNum, active && s.zoneNumActive]}>{z}</Text>
              <Text style={[s.zoneLabel, active && s.zoneLabelActive]}>
                {z === 1 ? 'F-L' : z === 2 ? 'F-R' :
                 z === 3 ? 'M-L' : z === 4 ? 'M-R' :
                 z === 5 ? 'B-L' : 'B-R'}
              </Text>
            </View>
          );
        })}
      </View>
      <Text style={s.netLabel}>▲ NET ▲</Text>
    </View>
  );
}

// ─── Timer bar ────────────────────────────────────────────────────────────────
function TimerBar({ running, delayMs }) {
  const anim = useRef(new Animated.Value(0)).current;
  const ref  = useRef(null);

  useEffect(() => {
    if (ref.current) ref.current.stop();
    anim.setValue(1);
    if (running && delayMs > 0) {
      ref.current = Animated.timing(anim, {
        toValue: 0,
        duration: delayMs,
        useNativeDriver: false,
      });
      ref.current.start();
    }
  }, [running, delayMs]);

  const width = anim.interpolate({
    inputRange: [0, 1], outputRange: ['0%', '100%'],
  });

  return (
    <View style={s.timerBg}>
      <Animated.View style={[s.timerFill, { width }]} />
    </View>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [model,       setModel]       = useState(null);
  const [source,      setSource]      = useState('No file loaded');
  const [loading,     setLoading]     = useState(false);
  const [mode,        setMode]        = useState('practice');
  const [phase,       setPhase]       = useState('idle'); // idle | rest | playing | done
  const [cues,        setCues]        = useState([]);
  const [cueIdx,      setCueIdx]      = useState(0);
  const [nRallies,    setNRallies]    = useState(0);
  const [rallyNum,    setRallyNum]    = useState(0);
  const [shotsDone,   setShotsDone]   = useState(0);
  const [currentCue,  setCurrentCue]  = useState(null);
  const [restCount,   setRestCount]   = useState(0);
  const [restInfo,    setRestInfo]    = useState('');
  const [timerMs,     setTimerMs]     = useState(0);

  const timerRef    = useRef(null);
  const restRef     = useRef(null);
  const cuesRef     = useRef([]);
  const idxRef      = useRef(0);
  const runningRef  = useRef(false);
  const shotsRef    = useRef(0);

  // Keep screen awake during session
  useEffect(() => {
    if (phase === 'playing' || phase === 'rest') {
      KeepAwake.activateKeepAwakeAsync();
    } else {
      KeepAwake.deactivateKeepAwake();
    }
  }, [phase]);

  // Load cached report on startup
  useEffect(() => {
    (async () => {
      try {
        const cached = await AsyncStorage.getItem('report');
        if (cached) {
          const data = JSON.parse(cached);
          setModel(parseReport(data));
          setSource(data.source ?? 'Cached report');
        }
      } catch {}
    })();
  }, []);

  // ── File loading ────────────────────────────────────────────────────────────
  const loadFile = useCallback(async () => {
    try {
      const res = await DocumentPicker.getDocumentAsync({
        type: 'application/json',
        copyToCacheDirectory: true,
      });
      if (res.canceled) return;

      setLoading(true);
      const text = await FileSystem.readAsStringAsync(res.assets[0].uri);
      const data = JSON.parse(text);

      if (!data.shot_frequency || !data.shots_by_zone) {
        Alert.alert('Invalid file', 'This JSON does not look like a trainer report.');
        setLoading(false);
        return;
      }

      await AsyncStorage.setItem('report', JSON.stringify(data));
      setModel(parseReport(data));
      setSource(data.source ?? res.assets[0].name);
      setLoading(false);
      Alert.alert('Loaded', `${data.shot_frequency?.length} shot types, `+
        `${data.shots_by_zone?.length} zones ready.`);
    } catch (e) {
      setLoading(false);
      Alert.alert('Error', String(e));
    }
  }, []);

  // ── Session control ─────────────────────────────────────────────────────────
  const stopAll = useCallback(() => {
    clearTimeout(timerRef.current);
    clearInterval(restRef.current);
    runningRef.current = false;
    Speech.stop();
  }, []);

  const startSession = useCallback(() => {
    if (!model) { Alert.alert('No data', 'Load a trainer report JSON first.'); return; }
    stopAll();

    const { cues: generated, nRallies: nr } = generateSession(model, mode);
    cuesRef.current = generated;
    idxRef.current  = 0;
    shotsRef.current = 0;
    runningRef.current = true;

    setCues(generated);
    setCueIdx(0);
    setNRallies(nr);
    setRallyNum(0);
    setShotsDone(0);
    setCurrentCue(null);
    setPhase('playing');

    scheduleNext(generated, 0, nr);
  }, [model, mode]);

  const pauseSession = useCallback(() => {
    stopAll();
    setPhase('idle');
  }, [stopAll]);

  const skipRally = useCallback(() => {
    stopAll();
    clearInterval(restRef.current);
    let i = idxRef.current + 1;
    while (i < cuesRef.current.length && !cuesRef.current[i].rallyStart) i++;
    if (i >= cuesRef.current.length) { finishSession(); return; }
    idxRef.current = i;
    setCueIdx(i);
    runningRef.current = true;
    scheduleNext(cuesRef.current, i, nRallies);
  }, [nRallies]);

  const finishSession = useCallback(() => {
    stopAll();
    setPhase('done');
  }, [stopAll]);

  const scheduleNext = useCallback((allCues, idx, nr) => {
    if (!runningRef.current || idx >= allCues.length) {
      finishSession(); return;
    }
    const cue = allCues[idx];

    if (cue.rallyStart && idx > 0) {
      // Show rest countdown
      const secs = Math.round(cue.delay_s);
      const paceLabel = cue.pace === 'fast' ? 'FAST' :
                        cue.pace === 'warmup' ? 'WARM-UP' : 'MODERATE';
      setRestCount(secs);
      setRestInfo(`Rally ${cue.rallyNum} — ${paceLabel} pace`);
      setPhase('rest');
      setCurrentCue(null);

      let rem = secs;
      restRef.current = setInterval(() => {
        rem--;
        setRestCount(Math.max(0, rem));
        if (rem <= 0) {
          clearInterval(restRef.current);
          if (runningRef.current) {
            setPhase('playing');
            playCue(allCues, idx, nr);
          }
        }
      }, 1000);
    } else {
      const ms = cue.delay_s * 1000;
      setTimerMs(ms);
      timerRef.current = setTimeout(() => {
        if (runningRef.current) playCue(allCues, idx, nr);
      }, ms);
    }
  }, []);

  const playCue = useCallback((allCues, idx, nr) => {
    if (!runningRef.current || idx >= allCues.length) return;
    const cue = allCues[idx];

    if (cue.rallyNum) setRallyNum(cue.rallyNum);
    setCurrentCue(cue);
    idxRef.current = idx + 1;
    shotsRef.current += 1;
    setCueIdx(idx + 1);
    setShotsDone(s => s + 1);

    Speech.speak(`${cue.zone} ${cue.shot}`, { rate: 1.2, pitch: 1.0 });

    scheduleNext(allCues, idx + 1, nr);
  }, [scheduleNext]);

  // ── Render helpers ──────────────────────────────────────────────────────────
  const isRunning   = phase === 'playing' || phase === 'rest';
  const progress    = cues.length ? cueIdx / cues.length : 0;
  const paceColor   = currentCue ? PACE_COLORS[currentCue.pace] : C.sub;
  const zoneArea    = currentCue?.zone >= 5 ? 'back' :
                      currentCue?.zone >= 3 ? 'mid' : 'front';
  const numColor    = zoneArea === 'back' ? C.orange :
                      zoneArea === 'mid'  ? C.yellow : C.green;

  return (
    <View style={s.root}>
      <StatusBar barStyle="light-content" backgroundColor={C.bg} />

      {/* ── Header ── */}
      <View style={s.header}>
        <Text style={s.logo}>AI COACH</Text>
        <Text style={s.sourceTag} numberOfLines={2}>{source}</Text>
      </View>

      {/* ── Mode bar ── */}
      <View style={s.modeBar}>
        {['warmup','practice'].map(m => (
          <TouchableOpacity key={m} style={[s.modeBtn, mode===m && s.modeBtnActive]}
            onPress={() => { if (!isRunning) setMode(m); }}>
            <Text style={[s.modeBtnTxt, mode===m && s.modeBtnTxtActive]}>
              {m === 'warmup' ? 'Warm-Up' : 'Practice'}
            </Text>
          </TouchableOpacity>
        ))}
        <TouchableOpacity style={s.loadBtn} onPress={loadFile} disabled={loading}>
          {loading
            ? <ActivityIndicator size="small" color={C.accent} />
            : <Text style={s.loadBtnTxt}>📂</Text>}
        </TouchableOpacity>
      </View>

      {/* ── Court ── */}
      <CourtDiagram activeZone={currentCue?.zone ?? null} />

      {/* ── Cue box ── */}
      <View style={s.cueBox}>
        <Text style={s.rallyInfo}>
          {phase === 'idle' ? 'Load a file & press Start' :
           phase === 'done' ? 'Session complete' :
           phase === 'rest' ? 'Rest' :
           `Rally ${rallyNum} / ${nRallies}`}
        </Text>
        <View style={[s.paceChip, { borderColor: paceColor }]}>
          <Text style={[s.paceChipTxt, { color: paceColor }]}>
            {currentCue?.pace?.toUpperCase() ?? '—'}
          </Text>
        </View>
        <Text style={[s.cueNum, { color: numColor }]}>
          {phase === 'rest' ? restCount : (currentCue?.zone ?? '—')}
        </Text>
        <Text style={s.cueShot}>
          {phase === 'rest' ? restInfo :
           currentCue?.shot?.toUpperCase() ?? 'SELECT MODE & START'}
        </Text>
        <Text style={s.cueSub}>
          {currentCue
            ? `${currentCue.zoneName}  ·  ${currentCue.delay_s}s`
            : phase === 'done' ? `${shotsDone} shots completed` : ''}
        </Text>
      </View>

      {/* ── Timer bar ── */}
      <TimerBar running={phase === 'playing'} delayMs={timerMs} />

      {/* ── Progress ── */}
      <View style={s.progRow}>
        <Text style={s.progTxt}>{cueIdx} / {cues.length}</Text>
        <View style={s.progBg}>
          <View style={[s.progFill, { width: `${progress * 100}%` }]} />
        </View>
        <Text style={s.progTxt}>{nRallies} rallies</Text>
      </View>

      {/* ── Controls ── */}
      <View style={s.btns}>
        <TouchableOpacity style={[s.btn, s.btnSm]}
          onPress={() => { stopAll(); setPhase('idle'); setCues([]); setCueIdx(0);
            setShotsDone(0); setCurrentCue(null); setRallyNum(0); }}>
          <Text style={s.btnTxt}>↺</Text>
        </TouchableOpacity>

        <TouchableOpacity style={[s.btn, s.btnMain,
          isRunning ? s.btnPause : s.btnStart]}
          onPress={isRunning ? pauseSession : startSession}>
          <Text style={[s.btnTxt, s.btnMainTxt]}>
            {isRunning ? 'Pause' : phase === 'done' ? 'Again' : 'Start'}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity style={[s.btn, s.btnSm]}
          onPress={skipRally} disabled={!isRunning}>
          <Text style={[s.btnTxt, !isRunning && { opacity: 0.3 }]}>⏭</Text>
        </TouchableOpacity>
      </View>

      {/* ── Done overlay ── */}
      {phase === 'done' && (
        <View style={s.overlay}>
          <Text style={s.overlayTitle}>Session Done</Text>
          <Text style={s.overlayStat}>
            {nRallies} rallies{'\n'}{shotsDone} shots{'\n'}
            Mode: {mode.toUpperCase()}
          </Text>
          <TouchableOpacity style={s.overlayBtn} onPress={startSession}>
            <Text style={s.overlayBtnTxt}>Go Again</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[s.overlayBtn, s.overlayBtnSecondary]}
            onPress={() => { setPhase('idle'); setCues([]); }}>
            <Text style={[s.overlayBtnTxt, { color: C.text }]}>Back</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────
const s = StyleSheet.create({
  root:          { flex:1, backgroundColor:C.bg, paddingTop:44,
                   paddingHorizontal:14, paddingBottom:20 },
  header:        { flexDirection:'row', justifyContent:'space-between',
                   alignItems:'center', marginBottom:10 },
  logo:          { fontFamily:'System', fontSize:17, fontWeight:'900',
                   letterSpacing:3, color:C.accent },
  sourceTag:     { fontSize:10, color:C.sub, textAlign:'right',
                   maxWidth:'58%', lineHeight:14 },

  modeBar:       { flexDirection:'row', gap:8, marginBottom:10 },
  modeBtn:       { flex:1, paddingVertical:9, borderRadius:8,
                   borderWidth:1.5, borderColor:C.border,
                   backgroundColor:C.surface, alignItems:'center' },
  modeBtnActive: { borderColor:C.accent },
  modeBtnTxt:    { fontSize:13, fontWeight:'700', color:C.sub,
                   letterSpacing:1, textTransform:'uppercase' },
  modeBtnTxtActive:{ color:C.accent },
  loadBtn:       { width:46, paddingVertical:9, borderRadius:8,
                   borderWidth:1.5, borderColor:C.border,
                   backgroundColor:C.surface, alignItems:'center',
                   justifyContent:'center' },
  loadBtnTxt:    { fontSize:18 },

  courtWrap:     { alignItems:'center', marginBottom:4 },
  court:         { width:160, height:206, flexDirection:'row',
                   flexWrap:'wrap', borderRadius:8,
                   borderWidth:1.5, borderColor:C.border,
                   overflow:'hidden', gap:1, backgroundColor:C.border },
  zoneCell:      { width:'50%', height:'33.33%',
                   backgroundColor:C.surface,
                   alignItems:'center', justifyContent:'center' },
  zoneNum:       { fontSize:28, fontWeight:'900', color:'rgba(255,255,255,0.14)' },
  zoneNumActive: { color:'#000' },
  zoneLabel:     { fontSize:8, fontWeight:'600', letterSpacing:1,
                   color:'rgba(255,255,255,0.12)', textTransform:'uppercase' },
  zoneLabelActive:{ color:'rgba(0,0,0,0.6)' },
  netLabel:      { fontSize:8, color:C.sub, letterSpacing:3, marginTop:3 },

  cueBox:        { backgroundColor:C.surface, borderWidth:1,
                   borderColor:C.border, borderRadius:12,
                   padding:12, alignItems:'center', marginBottom:6 },
  rallyInfo:     { fontSize:10, fontWeight:'700', letterSpacing:2,
                   color:C.sub, textTransform:'uppercase', marginBottom:3 },
  paceChip:      { borderWidth:1, borderRadius:20, paddingHorizontal:10,
                   paddingVertical:2, marginBottom:3 },
  paceChipTxt:   { fontSize:10, fontWeight:'700', letterSpacing:2,
                   textTransform:'uppercase' },
  cueNum:        { fontSize:84, fontWeight:'900', lineHeight:84,
                   letterSpacing:-4 },
  cueShot:       { fontSize:24, fontWeight:'700', color:C.text,
                   letterSpacing:1, textTransform:'uppercase', marginTop:2 },
  cueSub:        { fontSize:11, color:C.sub, marginTop:2, letterSpacing:0.5 },

  timerBg:       { height:3, backgroundColor:C.border, borderRadius:2,
                   overflow:'hidden', marginBottom:6 },
  timerFill:     { height:'100%', backgroundColor:C.accent, borderRadius:2 },

  progRow:       { flexDirection:'row', alignItems:'center',
                   gap:6, marginBottom:10 },
  progTxt:       { fontSize:10, color:C.sub, minWidth:40 },
  progBg:        { flex:1, height:3, backgroundColor:C.border, borderRadius:2 },
  progFill:      { height:'100%', backgroundColor:C.sub, borderRadius:2 },

  btns:          { flexDirection:'row', gap:8 },
  btn:           { borderRadius:10, alignItems:'center',
                   justifyContent:'center', paddingVertical:16 },
  btnSm:         { width:50, backgroundColor:C.surface,
                   borderWidth:1, borderColor:C.border },
  btnMain:       { flex:1 },
  btnStart:      { backgroundColor:C.green },
  btnPause:      { backgroundColor:C.orange },
  btnTxt:        { fontSize:17, fontWeight:'900', color:C.text,
                   letterSpacing:0.5 },
  btnMainTxt:    { color:'#001a0d', fontSize:18 },

  overlay:       { position:'absolute', inset:0, backgroundColor:'rgba(4,8,14,0.96)',
                   alignItems:'center', justifyContent:'center',
                   padding:32, gap:14 },
  overlayTitle:  { fontSize:38, fontWeight:'900', color:C.green,
                   letterSpacing:4, textTransform:'uppercase' },
  overlayStat:   { fontSize:15, color:C.sub, textAlign:'center', lineHeight:28 },
  overlayBtn:    { paddingVertical:14, paddingHorizontal:44,
                   backgroundColor:C.accent, borderRadius:10 },
  overlayBtnSecondary: { backgroundColor:C.surface,
                          borderWidth:1, borderColor:C.border },
  overlayBtnTxt: { fontSize:16, fontWeight:'900', color:'#001a22',
                   letterSpacing:1, textTransform:'uppercase' },
});
