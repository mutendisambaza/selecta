# Selecta

**A DJ's assistant and coach.** Point it at a folder of tracks, get back a sequenced set plan and Rekordbox-ready cue points — in seconds, not hours.

---

## What it does

Selecta solves the cold-start problem every DJ knows: you have 200 tracks, a gig in 3 hours, and building a quality set from scratch takes serious time. Selecta does the analytical groundwork so you start from a strong foundation rather than a blank page.

Given a folder of audio files it produces two things:

**Set plan** — a numbered sequence of tracks ordered to take a listener on a journey (warm-up → build → peak → resolve), with per-transition coaching notes:

```
19. STANDARD — 11B | 118.0 BPM | energy 0.76 (target 0.76)
    ↳ from 11B to 11B | BPM +0.0% | bring in at 0:30 | swap bass at 0:30 | mix score 83

20. Buya (Official Audio) — 10B | 112.7 BPM | energy 0.66 (target 0.84)
    ↳ from 11B to 10B | BPM -4.5% | bring in at 1:18 | swap bass at 1:18 | mix score 74
```

**Rekordbox XML** — phrase boundaries and key moments exported as cue points that appear on the Rekordbox waveform the moment you import the file. No manual cue-marking needed before a session.

---

## How it works

### Feature extraction
Selecta analyses each track once and caches the results. For every file it extracts:

- **BPM and beat grid** — via librosa's beat tracker, windowed to the track's body (skipping intros and outros) to avoid amapiano/gqom breakdown sections corrupting the estimate
- **Key (Camelot wheel)** — Krumhansl-Schmuckler key profiles over chroma-CQT
- **Intensity** — a three-feature composite designed to survive heavy mastering:
  - *Spectral centroid* (0.50 weight) — where the center of mass sits spectrally; warm/sub-heavy intro tracks score low, bright/stab-heavy peak tracks score high
  - *Beat-synchronized HF fraction* (0.35) — proportion of energy above 4 kHz at detected beat positions; captures punch distribution without being fooled by overall loudness
  - *Dynamic complexity* (0.15) — mean absolute deviation of per-frame loudness; validated empirically (r = +0.33) on a 30-track amapiano crate
- **Phrase regions** — 16-bar phrase grid with labels (intro, build, drop, breakdown, outro)
- **Spectral bands** — per-phrase sub/mid/high energy profiles used to score timbral fit between transitions

### Sequencing
The sequencer reads cached features and solves a path-finding problem: order the tracks so that:

1. Consecutive transitions score well (harmonically compatible keys, small BPM delta, smooth energy/spectral flow)
2. The track's position in the set follows a target energy arc

Two arc profiles are built in:
- **experiential** — a full journey arc (soft start → building → deliberate dip → peak → resolve); suitable for a headline set
- **just_mix** — flat arc that prioritises smooth transitions over narrative shape

The algorithm seeds greedily from the best arc-start candidate, then refines the ordering with 2-opt swaps. Because heavily mastered tracks (especially amapiano) cluster in absolute intensity, a rank/percentile mapping is applied first so the full [0.0, 1.0] range is always available for arc matching.

### Scoring
Each transition is scored across five dimensions:

| Dimension | What it measures |
|-----------|-----------------|
| Harmonic (30%) | Camelot key compatibility — same, adjacent, relative, energy boost, or distant |
| Tempo (25%) | BPM stretch required; 6% window is seamless, beyond that costs points |
| Energy (20%) | How close the incoming track's intensity is to the arc target at that position |
| Spectral (15%) | Timbral overlap between the outgoing track's outro and the incoming track's intro |
| Phrase (10%) | Whether both tracks have clean outro/intro phrase markers available |

The per-dimension breakdown is in the JSON output (`set-plan.json`), available for deeper analysis or UI display.

---

## Setup

Selecta uses a conda environment to manage the Python version, librosa, and the native audio stack.

```bash
# create the environment (first time)
conda env create -f environment.yml

# activate
conda activate selecta

# install selecta itself (editable)
pip install -e .
```

Requires conda (Miniconda or Anaconda). On Apple Silicon (osx-arm64) the full stack installs from conda-forge with no extra steps.

---

## Usage

```bash
# 1. Analyse a folder — extracts and caches features for every audio file
selecta analyze /path/to/tracks

# 2. Export Rekordbox cue points (import the XML via File → Import Playlist in Rekordbox)
selecta cues /path/to/tracks

# 3. Generate a sequenced set plan
selecta sequence /path/to/tracks

# Optional flags
--db /custom/path/features.db    # use a specific feature cache
--profile just_mix               # arc profile (default: experiential)
```

Supported formats: `.mp3`, `.wav`, `.aiff`, `.flac`, `.m4a`, `.ogg`

The feature cache is content-hash based — re-analysing a folder only processes new or changed files.

---

## Supported genres

| Tier | Genres | Notes |
|------|--------|-------|
| 1 | House, Tech House, Afro House / Afrotech | Clean 4/4, high-confidence BPM and phrase detection |
| 2 | Amapiano, Gqom | Syncopated rhythms, half-time BPM ambiguity; phrase confidence is lower but detection works |

---

## Output files

| File | Contents |
|------|----------|
| `set-plan.md` | Human-readable numbered set plan with transition coaching notes |
| `set-plan.json` | Full structured data — intensities, arc targets, per-dimension scores, phrase cues |
| `rekordbox.xml` | Import this into Rekordbox via File → Import Playlist to load all phrase cues |

---

## Tuning

`config.toml` exposes the scoring weights and arc profile control points. All values are documented inline. Changes take effect on the next `sequence` run without needing to re-analyse.

---

## What Selecta is not (v1)

- It does not render audio or create crossfades
- It does not write to the Rekordbox library database directly (import-based only)
- It does not run in real time alongside a DJ set
- It does not use machine learning — every decision it makes is explainable

These are intentional constraints for v1. The explainability is the point: Selecta is a coach, not a black box.
