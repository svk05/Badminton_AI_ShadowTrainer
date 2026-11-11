# Badminton AI Coach — Android App

Standalone Android APK. No PC needed on court.
Built with Expo (React Native) — APK compiled in Expo's cloud.

---

## SETUP (one time, ~15 minutes)

### 1. Install Node.js
Download from: https://nodejs.org  (LTS version)
Run the installer. Verify with:
```
node --version
```

### 2. Install Expo CLI + EAS CLI
```
npm install -g expo-cli eas-cli
```

### 3. Create a free Expo account
https://expo.dev  → Sign Up (free, no credit card)

### 4. Copy project files
Put all files from this folder into a single directory, e.g.:
```
C:\SVK\PERSONAL_PROJECTS\AI_COACH_APP\
```

Your structure should look like:
```
AI_COACH_APP/
  App.js
  app.json
  eas.json
  package.json
  babel.config.js
  src/
    rallyEngine.js
  assets/
    icon.png       ← add any 1024x1024 PNG (can be a placeholder)
    splash.png     ← add any PNG (can be same as icon)
```

### 5. Install dependencies
```
cd C:\SVK\PERSONAL_PROJECTS\AI_COACH_APP
npm install
```

### 6. Log into Expo
```
eas login
```

---

## BUILD THE APK (cloud build, no Android Studio needed)

```
cd C:\SVK\PERSONAL_PROJECTS\AI_COACH_APP
eas build --platform android --profile preview
```

- First build takes ~10-15 minutes (runs on Expo's servers)
- You'll get a download link in the terminal AND at https://expo.dev/builds
- Download the .apk file

---

## INSTALL ON PHONE

1. Transfer the .apk to your phone (USB cable or Google Drive)
2. On Android: Settings → Security → Install unknown apps → Allow
3. Tap the .apk file to install
4. Open "AI Coach"

---

## USING THE APP

### First time: load your trainer report
1. Copy your `rally_cues.json` to your phone (Downloads folder or Google Drive)
2. Open AI Coach → tap 📂 → select the JSON file
3. The app remembers it — you don't need to reload next time

### Training session
- **Warm-Up**: 5 rallies, 8-10 shots, 1.45× slower pace. 5-6s rest between rallies.
- **Practice**: 8 rallies alternating moderate and fast pace, 8-15 shots each.
  3-4s rest between rallies with countdown.

### Controls
- **Start / Pause** — begin or pause session
- **↺** — reset session
- **⏭** — skip to next rally

### Physics timing (what makes it realistic)
- Smash → net-shot: ~1.3s  (opponent scrambled to lift)
- Smash → front zone: ~1.6s
- Back → front (non-smash): ~2.33s
- Drop/clear: extra recovery time built in
- Net-shots/drives: fast, versatile

---

## UPDATING DATA

When you analyse new pro videos and generate new JSON reports:
1. Copy the new JSON to your phone
2. Open AI Coach → tap 📂 → select the new file
The app updates immediately and remembers the new data.

---

## REBUILDING (if you change the code)

```
eas build --platform android --profile preview
```

Same process — Expo builds in the cloud, you download the new APK.
