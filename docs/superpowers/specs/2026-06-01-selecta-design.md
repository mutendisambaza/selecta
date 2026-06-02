# Selecta — DJ Auto-Mix Setlist Planner

**Date:** 2026-06-01
**Status:** Design approved (pending spec review)

## 1. Purpose

Point Selecta at a folder of electronic (4-on-the-floor) tracks and get back two
things, both built on the same per-track analysis:

1. **A sequenced DJ setlist with per-transition instructions** — a *plan*, not a
   rendered audio file. Selecta scores how well every pair would mix using a
   transparent heuristic and orders the library into a coherent set that follows
   an energy arc.
2. **Preset phrase cue points exported to Rekordbox** — every track's detected
   phrase boundaries (intro / build / drop / breakdown / outro) written as
   labelled cues that show up on the waveform in Rekordbox, so the prep work of
   marking phrases is done automatically across the whole library.

Selecta answers the questions a DJ asks when prepping and building a set: *"where
are the phrases in this track, what plays well after what, where do I bring the
next one in, and does the whole thing flow?"*

### Non-goals (v1)
- Does **not** render or export mixed audio (no time-stretch/EQ/crossfade output).
- Does **not** use machine learning or audio embeddings for scoring.
- Not aimed at live drums, heavy tempo drift, or odd time signatures.

## 2. Scope & Assumptions

- **Music:** electronic, 4/4, steady tempo, clean 8/16/32-bar phrasing.
- **Outputs:** Markdown plan + JSON (the setlist); `rekordbox.xml` (the cues).
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
        ┌──────────────────────────────────────┼───────────────────────┐
        ▼                                       │                       ▼
  ┌──────────────┐   pairwise    ┌────────────┐ │           ┌────────────────────┐
  │Pairwise Score│ ────────────► │ Sequencer  │ │           │   Cue Exporter     │
  │  (heuristic) │   matrix      │ (greedy+2opt│ │           │ phrase map → cues  │
  └──────────────┘               │  under arc)│ │           │ → rekordbox.xml    │
                                 └─────┬──────┘ │           └────────────────────┘
                                       ▼        │
                              ┌───────────────┐ │
                              │ Plan Renderer │◄┘
                              │ (md + json)   │
                              └───────────────┘
```

CLI exposes three commands: `selecta analyze <folder>` (extract + cache),
`selecta sequence <folder> [--arc <profile>]` (the setlist plan), and
`selecta cues <folder> [--out rekordbox.xml]` (the Rekordbox phrase cues).

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

### 4.6 Cue Exporter
**Job:** `TrackFeatures` (one or many) → a `rekordbox.xml` carrying phrase cue
points that display on the waveform in Rekordbox. Consumes the phrase map from
§4.1 — **no change to extraction**.

- **Format:** Rekordbox XML (`DJ_PLAYLISTS` → `COLLECTION` → `TRACK` → repeated
  `<POSITION_MARK>`). Import-based: the user points Rekordbox at the file
  (Preferences → Advanced → Database → rekordbox xml); cues arrive when tracks
  are dragged into the collection. **Never writes the live Rekordbox database**,
  so there is no corruption risk.
- **Cue mapping:**
  - Every phrase boundary → a **memory cue** (`Num = -1`), labelled by region
    (Intro / Build / Drop / Breakdown / Outro). Unlimited, full coverage.
  - The key moments — first drop, main breakdown, outro start (and intro start) —
    are **promoted to hot cues** in slots A–H (`Num = 0..7`) for instant jumping.
    Capped at 8; if more key moments than slots, keep the highest-priority ones
    and leave the rest as memory cues.
  - Cue positions are **absolute seconds**, independent of Rekordbox's beatgrid.
- **Confidence:** low-confidence downbeat/phrase detections are labelled (e.g.
  `Drop?`) so the user knows to eyeball them.
- **Track identity:** `<TRACK>` `Location` is the file URI; existing tags
  (title/artist/BPM/key) are written so Rekordbox matches to the right file.

**Interface:** `export_cues(tracks: list[TrackFeatures], out_path: str) -> None`
**Tested by:** golden-file test on emitted XML for a fixture track; schema-validate
that Rekordbox-required attributes are present.

### 4.7 CLI
- `selecta analyze <folder>` — extract + cache features (skips unchanged files).
- `selecta sequence <folder> [--arc <profile>] [--out <path>]` — score, order,
  render plan. Assumes analyse has run (or runs it implicitly for new files).
- `selecta cues <folder> [--out rekordbox.xml]` — export Rekordbox phrase cues
  for every analysed track. Also runs analyse implicitly for new files.

## 5. Data Flow

1. `analyze` walks the folder, hashes each file, extracts features for new/changed
   files, upserts into the store.
2. `sequence` loads all features, computes the pairwise matrix, runs the sequencer
   under the chosen arc, renders the plan to Markdown + JSON.
3. `cues` loads all features, maps each track's phrase map to memory + hot cues,
   and writes a single `rekordbox.xml` for the library.

The two outputs (§4.5 plan, §4.6 cues) are independent consumers of the same
cached features — either can run without the other.

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
- **Cue Exporter:** golden-file test on emitted `rekordbox.xml`; assert required
  `POSITION_MARK` attributes (Name, Type, Start, Num) and that hot-cue slots stay
  within A–H. Manual check: import once into Rekordbox, confirm cues land on the
  waveform at the right phrases.
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
- **Direct Rekordbox write** (`pyrekordbox`) — a second Cue Exporter backend that
  writes cues straight into the Rekordbox database + ANLZ files (no import step).
  Same `export_cues()` interface; gated behind backups because it mutates the live
  collection and is sensitive to Pioneer DB-format changes.
- **Other DJ software targets** — Serato (GEOB ID3 cue frames) / Traktor (NML)
  exporters behind the same interface.

## 9. Tech Stack

- Python 3.12 (dedicated venv).
- librosa, numpy, scipy, soundfile.
- SQLite (stdlib `sqlite3`).
- TOML config (stdlib `tomllib`).
- CLI via `argparse` or `click`.
- Rekordbox XML via stdlib `xml.etree.ElementTree` (no new dep). `pyrekordbox`
  only if/when the future direct-write backend is built.
- ffmpeg + rubberband (system, already installed).
