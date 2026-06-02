# Selecta — Session Checkpoint (2026-06-02)

## Current state

**HEAD:** `82ce156` — all work committed and pushed to `git@github-personal:mutendisambaza/selecta.git`  
**Tests:** 40/40 passing  
**Last full run:** MUZE ROOF 30-track crate — 0 failures, plan produced, rekordbox.xml generated

---

## What was built this session (on top of the MVP)

### R3 — Spectral brightness energy composite (`9523067`)
Replaced the old `mean_RMS + onset_density` intensity (clustered 0.56–0.69 on amapiano) with a three-feature composite that survives mastering normalization:

| Feature | Weight | Rationale |
|---|---|---|
| Spectral centroid, normalized to 4 kHz | 0.50 | Spectral shape is loudness-invariant; warm/sub-heavy intro = low, bright stabs/hats = high |
| Beat-synchronized HF fraction (energy above 4 kHz at beat positions) | 0.35 | Rhythm-weighted; measures where punch lands spectrally, not time-average |
| DynamicComplexity (mean abs deviation of per-frame loudness, dB) | 0.15 | Empirically validated on MUZE ROOF: r=+0.33, use DC_norm (not inverted) |

**Result:** spread 0.000–1.000 on the crate (was 0.56–0.69). Rank/percentile calibration kept in sequencer as final layer — endorsed by research (Bittner et al., ISMIR 2017 doesn't even do arc ordering).

### BPM windowing fix (`d5866f5`)
`librosa.beat.beat_track` on full-length amapiano tracks gave corrupt BPM due to breakdowns/outros. Fix: `_estimate_bpm()` runs `onset_strength` + `feature.tempo` (autocorrelation) on a `[15s, 105s]` window of the audio. Beat *grid* still uses the full track (needed for phrase detection). Short tracks (<45 s) fall back to beat interval mean.

**Result:**
- Awe Mah: 90.3 → **117.2 BPM** (Rekordbox truth: 118) ✓
- Vele Kwaphoseka: 168.9 → **125.0 BPM** (Rekordbox truth: 124) ✓

### README (`82ce156`)
First real README — covers what Selecta is, feature extraction, sequencing algorithm, scoring table, setup, usage, supported genres, output files, tuning, and v1 scope limits.

---

## Deep research findings (stored, not yet fully implemented)

The deep-research workflow (105 agents) confirmed:

1. **Root cause of clustering** — intra-track-relative RMS is structurally incapable of cross-track discrimination (EURASIP 2018, 3-0 adversarial vote). Our diagnosis was exactly right.
2. **Best features** — spectral shape (centroid, HF band ratios) survive mastering normalization. Implemented.
3. **DynamicComplexity direction** — empirically validated as positive (r=+0.33). Implemented.
4. **Rank mapping is correct** — Bittner et al. ISMIR 2017 (Spotify Research) uses no arc ordering at all. Rank is state-of-the-art.
5. **Open research question** — genre-aware reference calibration (z-score against a corpus of known DJ sets). Not implemented; deferred.

---

## Known issues / next priority order

### 1. Key clustering (HIGH) — 9/30 tracks read `7A`
The Krumhansl-Schmuckler detection is over-assigning `7A` (Bb minor). This hurts the harmonic scorer because several good transitions are being marked "distant" when the true keys are compatible. The fix is in `src/selecta/features/key.py` — likely a confidence threshold issue or tonic-detection bias. Rekordbox ground truth available:
- Awe Mah: `6A` (detected `7A`)
- Vele Kwaphoseka: `2A` (detected `12A`)

### 2. Score breakdown in plan.md (MEDIUM)
The text plan shows "mix score 83" but not the per-dimension breakdown (harmonic / tempo / energy / spectral / phrase). The data is already in `set-plan.json`. Exposing it in the markdown would make the "coach" framing real — a DJ can see *why* a transition scores well or poorly. Fix is in `src/selecta/render/plan.py`.

### 3. GUITARS ON 3STEP at 93.8 BPM (LOW — needs ground truth)
Still detecting 93.8 BPM. No Rekordbox ground truth provided yet. 3Step genre sometimes runs at ~65–70 BPM (half-time feel); 93.8 may be correct for this track. Get Rekordbox value before touching it.

### 4. Swap-bass cue coincides with bring-in cue (LOW)
In the current plan renderer, "bring in at X" and "swap bass at X" often show the same timestamp. Swap-bass should be expressed relative to groove entry, not the absolute cue time. Fix is in `src/selecta/render/plan.py`.

---

## File map (source)

```
src/selecta/
  features/
    energy.py       ← spectral brightness composite (R3 + DynamicComplexity)
    tempo.py        ← windowed BPM estimation (BPM fix)
    key.py          ← K-S key detection → Camelot (key clustering issue lives here)
    phrasing.py     ← 16-bar phrase grid
    extractor.py    ← orchestrates all features + mutagen tags
    types.py        ← frozen dataclasses (TrackFeatures, Beat, PhraseRegion)
    spectral.py     ← per-phrase band profiles
  sequencing/
    sequencer.py    ← greedy + 2-opt with rank mapping
    arc.py          ← energy arc interpolation
  scoring/
    scorer.py       ← pairwise transition scorer (5 dimensions)
    camelot.py      ← Camelot wheel relation + mood
  store/
    db.py           ← SQLite feature cache
    hashing.py      ← SHA-1 content hash for cache invalidation
  export/
    rekordbox.py    ← Rekordbox XML writer
  render/
    plan.py         ← set plan markdown + JSON renderer
  cli.py            ← analyze / cues / sequence commands
  pipeline.py       ← folder analysis helper
config.toml         ← scoring weights + arc control points
environment.yml     ← conda env (Python 3.12, librosa, conda-forge)
```

---

## How to resume

```bash
cd ~/Projects/selecta
conda activate selecta
git log --oneline -5       # confirm you're on HEAD 82ce156

# Re-run MUZE ROOF if you want fresh numbers (clear cache first):
rm /tmp/selecta-muze.db
selecta analyze "/Users/baz/Desktop/USB INTERIM/SETS/MUZE ROOF" --db /tmp/selecta-muze.db
selecta sequence "/Users/baz/Desktop/USB INTERIM/SETS/MUZE ROOF" --db /tmp/selecta-muze.db
```

Start with **key clustering** (`key.py`) — that's the highest-impact fix remaining.
