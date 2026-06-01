# Selecta — DJ Auto-Mix Setlist Planner

**Date:** 2026-06-01
**Status:** Design approved (pending spec review)

## 1. Purpose

Point Selecta at a folder of electronic (4-on-the-floor) tracks and get back a
**sequenced DJ setlist with per-transition instructions** — a *plan*, not a
rendered audio file. It analyses each track, scores how well every pair would
mix using a transparent heuristic, and orders the library into a coherent set
that follows an energy arc.

Selecta answers the question a DJ asks when building a set: *"what plays well
after what, where do I bring the next one in, and does the whole thing flow?"*

### Non-goals (v1)
- Does **not** render or export mixed audio (no time-stretch/EQ/crossfade output).
- Does **not** use machine learning or audio embeddings for scoring.
- Not aimed at live drums, heavy tempo drift, or odd time signatures.

## 2. Scope & Assumptions

- **Music:** electronic, 4/4, steady tempo, clean 8/16/32-bar phrasing.
- **Output:** Markdown plan (human-readable) + JSON (machine-readable).
- **Scoring brain:** hand-tuned weighted heuristic with a per-dimension breakdown.
- **Platform:** macOS (ARM). Python **3.12** venv (system 3.14 is too new for the
  audio stack). `ffmpeg` + `rubberband` already present via Homebrew.
- **DSP:** **librosa-first**; add `madmom` only if downbeat/phrasing quality is
  insufficient.

## 3. Architecture

Two-phase pipeline so the slow part (feature extraction) runs once and is cached,
letting the scoring/sequencing logic be iterated on instantly.

```
                 ┌──────────────────┐
  audio files ─► │ Feature Extractor│ ─► TrackFeatures
                 └──────────────────┘          │
                                               ▼
                                       ┌──────────────┐
                                       │ Feature Store│  (SQLite, hash-keyed)
                                       └──────────────┘
                                               │
        ┌──────────────────────────────────────┘
        ▼
  ┌──────────────┐   pairwise    ┌────────────┐   ordering   ┌───────────────┐
  │Pairwise Score│ ────────────► │ Sequencer  │ ───────────► │ Plan Renderer │
  │  (heuristic) │   matrix      │ (greedy+2opt│              │ (md + json)   │
  └──────────────┘               │  under arc)│              └───────────────┘
                                 └────────────┘
```

CLI exposes the two phases: `selecta analyze <folder>` and
`selecta sequence <folder> [--arc <profile>]`.

## 4. Components

Each component has one job, a defined interface, and is testable in isolation.

### 4.1 Feature Extractor
**Job:** one audio file → one `TrackFeatures` record. Deterministic, pure DSP.

Extracts:
- **Tempo + beat grid** — BPM and downbeat-aligned beat timestamps (librosa beat
  tracking; downbeat inferred from 4/4 assumption, refined by madmom later if needed).
- **Key** → Camelot notation — chroma features + Krumhansl-Schmuckler key profiles.
- **Energy curve** — RMS (and optionally LUFS) sampled over time, plus a single
  summary "intensity" scalar for arc placement.
- **Spectral profile** — energy per frequency band (low / low-mid / high-mid /
  high), computed **per section** (intro / body / outro), not just globally.
- **Phrase map** — downbeats grouped into 8/16/32-bar phrases; intro / outro /
  breakdown regions tagged. These are the *mixable zones*.

**Interface:** `extract(path: str) -> TrackFeatures`
**Depends on:** librosa, numpy, soundfile/ffmpeg for decoding.
**Tested by:** tracks with known BPM/key; assert detected values within tolerance.

### 4.2 Feature Store
**Job:** persist and retrieve `TrackFeatures`, skip already-analysed files.

- SQLite, one row per track. Scalar features as columns; curves/phrase maps as
  JSON blobs.
- Keyed by **file path + audio content hash** so moved/renamed files re-key
  correctly and unchanged files are skipped on re-run.
- Stores the **full rich feature set** even though the heuristic won't consume all
  of it — this table is the training dataset for the future ML scorer.

**Interface:** `upsert(features)`, `get(path)`, `all()`, `needs_analysis(path)`.

### 4.3 Pairwise Scorer (heuristic — the heart)
**Job:** `(TrackA, TrackB)` → `TransitionScore` (0–100) **with a per-dimension
breakdown**. Directional (A→B may differ from B→A).

Weighted blend of dimensions:
- **Harmonic** — Camelot adjacency. Same key = best; ±1 on wheel or
  relative major/minor = good; energy-boost moves = ok; distant = clash.
- **Tempo** — % stretch to match B to A; small gaps best; half/double-time
  treated as compatible.
- **Energy** — controlled delta in the direction the arc wants at that point.
- **Spectral fit** — during the blend, do A's outro bands and B's intro bands
  sit **complementary** (clean) vs overlapping in the same band (muddy)?
- **Phrase fit** — does A have a clean outro phrase and B a clean intro phrase
  long enough to blend over?

Weights live in a **config file** (`config.toml`) tuned by ear. The scorer always
returns the breakdown, never a bare number, so decisions are explainable.

**Interface:** `score(a: TrackFeatures, b: TrackFeatures, arc_pos: float) -> TransitionScore`

### 4.4 Sequencer
**Job:** library of `TrackFeatures` → an ordered setlist.

- Build the pairwise score matrix over all tracks.
- Find an ordering via **greedy nearest-neighbour → 2-opt refinement**.
- Constrain by a **target energy arc** (configurable profile, e.g.
  `warmup → build → peak → cooldown`): each track's intensity should match its
  position on the arc, and the objective combines transition quality + arc
  adherence.
- Each track used at most once (TSP-flavoured path, not a cycle).

**Interface:** `sequence(tracks, arc_profile) -> list[OrderedTrack]`

### 4.5 Plan Renderer
**Job:** ordered setlist → human + machine output.

- **Markdown:** track order with key/BPM, and per-transition instructions, e.g.
  *"Bring B in at bar 96 of A (3:12). Both 8A. +1.2% on B. Swap bass at bar 104."*
- **JSON:** same data, structured, for tooling / future audio-render layer.

Transition points come from aligning A's outro phrase boundary with B's intro
phrase boundary on the downbeat grid.

### 4.6 CLI
- `selecta analyze <folder>` — extract + cache features (skips unchanged files).
- `selecta sequence <folder> [--arc <profile>] [--out <path>]` — score, order,
  render plan. Assumes analyse has run (or runs it implicitly for new files).

## 5. Data Flow

1. `analyze` walks the folder, hashes each file, extracts features for new/changed
   files, upserts into the store.
2. `sequence` loads all features, computes the pairwise matrix, runs the sequencer
   under the chosen arc, renders the plan to Markdown + JSON.

## 6. Error Handling

- **Undecodable / corrupt file** → log a warning, skip the track, continue.
- **Low-confidence key/BPM** → store with a confidence flag; renderer marks
  uncertain transitions so the user knows to trust their ears.
- **Library too small to fill arc** → produce the best partial set, warn.
- **No good transition for a track** → sequencer may strand it; report stranded
  tracks rather than forcing a bad mix.

## 7. Testing Strategy

- **Feature Extractor:** known-BPM/known-key reference tracks → assert within
  tolerance; synthetic click-tracks for beat-grid accuracy.
- **Pairwise Scorer:** unit tests on Camelot adjacency table, tempo-gap math,
  spectral-overlap logic with crafted feature fixtures.
- **Sequencer:** small synthetic libraries with a known-optimal order; assert the
  arc constraint is respected and each track used once.
- **Plan Renderer:** golden-file tests on Markdown/JSON output.
- **Tuning loop (not a test):** a scratch notebook to audition scoring weights
  against a real library by ear.

## 8. Future Work (designed-for, NOT built in v1)

The interfaces above are stable so these slot in without rework:
- **ML learned scorer** — replaces §4.3's formula with a model trained on the
  §4.2 feature store + real DJ tracklists/recorded sets. Same `score()` interface.
- **Audio embeddings** (CLAP / OpenL3) — an added "vibe similarity" dimension in
  §4.3's blend.
- **Audio render layer** — a plan-executor that time-stretches, EQs, and
  crossfades the planned set into a finished audio file (rubberband/ffmpeg already
  available).
- **madmom upgrade** — swap in for downbeat/phrase detection if librosa's
  phrasing proves insufficient.

## 9. Tech Stack

- Python 3.12 (dedicated venv).
- librosa, numpy, scipy, soundfile.
- SQLite (stdlib `sqlite3`).
- TOML config (stdlib `tomllib`).
- CLI via `argparse` or `click`.
- ffmpeg + rubberband (system, already installed).
