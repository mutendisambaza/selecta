# Selecta — A Music Information Retrieval (MIR) Assistant for DJs

**Date:** 2026-06-01
**Status:** Design approved (pending spec review)

## 1. Purpose

Selecta is, at its core, a **Music Information Retrieval (MIR)** application built
for DJs. MIR is the field of extracting musical meaning from audio signals —
tempo, beat grid, key, energy, timbre/spectral content, and structure/phrasing.
Selecta runs that MIR analysis over a DJ's library and turns the extracted
features into practical prep and decision tooling: phrase cues, mix-compatibility
scoring, and set sequencing. Everything downstream is an application of the same
MIR feature layer.

Point Selecta at a folder of tracks and get back two things, both built on the
same per-track MIR analysis:

1. **A sequenced DJ set that takes the listener on a journey** — plus the key
   insights behind it. Not just "tracks that mix," but an *order with narrative
   shape* (ethereal/melodic start → build → peak → resolve), with per-transition
   instructions and the per-dimension reasoning for each placement. A *plan*, not
   a rendered audio file. The ambition is **one-shot**: point Selecta at a folder
   and get back a set good enough to play with light refinement — precision out of
   the gate, sharpened over time.
2. **Preset phrase cue points exported to Rekordbox** — every track's detected
   phrase boundaries (intro / build / drop / breakdown / outro) written as
   labelled cues that show up on the waveform in Rekordbox, so the prep work of
   marking phrases is done automatically across the whole library.

Selecta answers the questions a DJ asks when prepping and building a set: *"where
are the phrases in this track, what plays well after what, where do I bring the
next one in, and does the whole thing flow?"*

### Positioning
Selecta is a **DJ's assistant and coach** — it does not replace the DJ's taste,
it removes the cold-start problem so you never work from a blank library:
- **Baselining / foundation** — phrase cues plus a suggested setlist give you a
  *starting skeleton* to refine, not a blank page.
- **On-the-fly** — because the prep is done (cues on the waveform, "what mixes
  next" pre-computed), in-the-moment decisions are faster and more confident.
- **Coach** — the per-dimension score breakdown explains *why* a transition works
  or clashes, teaching the reasoning rather than handing down a verdict.

This positioning reinforces the **explainable heuristic** choice (§4.3): a
transparent, decomposable score can coach; an opaque "82/100" cannot — which is
why ML scoring stays a future layer, not the v1 brain.

> **Scope note:** v1 is *prep that pays off live* — offline/batch analysis whose
> value shows up in the moment. A real **real-time live mode** (listen to what's
> playing, suggest next tracks on the fly) is captured as future work (§8), not
> built in v1.

### Non-goals (v1)
- Does **not** render or export mixed audio (no time-stretch/EQ/crossfade output).
- Does **not** use machine learning or audio embeddings for scoring.
- Not aimed at live drums, heavy tempo drift, or odd time signatures.

## 2. Scope & Assumptions

- **Music:** DJ-oriented electronic with a South-African lean — **house, tech
  house, Afro house / Afrotech, gqom, amapiano**. These split into two tiers for
  analysis confidence:
  - **Tier 1 — 4-on-the-floor (high confidence):** house, tech, Afrotech. Steady
    tempo, strong 4/4, clean 8/16/32-bar phrasing. Beat grid, downbeat and phrase
    detection are reliable here; this is the primary v1 target.
  - **Tier 2 — broken-beat (supported, lower confidence):** **gqom** (sparse,
    syncopated, frequently *not* four-on-the-floor) and **amapiano** (~112–115 BPM,
    log-drum syncopation, prone to half/double-time BPM mis-detection). These
    break the steady-4/4 assumption, so downbeat/phrase detection is harder and
    BPM can be mis-octaved. v1 still handles them but **flags lower confidence**,
    and they are the most likely reason to pull the `madmom` upgrade forward (§8).
- **Outputs:** Markdown plan + JSON (the setlist); `rekordbox.xml` (the cues).
- **Scoring brain:** hand-tuned weighted heuristic with a per-dimension breakdown.
- **Platform / environment:** macOS (ARM), managed with **conda** (miniforge).
  A pinned `environment.yml` provides an isolated Python **3.12** plus the native
  audio stack — sidestepping system Python 3.14 (too new for the audio libs) and
  the painful from-source pip builds. See §9.
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
- **Harmonic** — Camelot adjacency, encoding real harmonic-mixing theory.
  Same key = perfect (100%); **±1 on the wheel = the workhorse move** (one note
  changes, listener barely notices); relative major/minor (same number, swap
  letter) = a **mood shift** — minor→major *lifts/brightens*, major→minor
  *darkens/drives*; the +7 "energy-boost" jump = usable; distant keys = clash.
  The scorer is **direction-aware**: which harmonic move is "good" depends on
  whether the journey wants to lift, drive, or hold steady at that point.
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
**Job:** library of `TrackFeatures` → an ordered setlist that takes the listener
on a **journey**. This is the headline of Selecta — the part that does the work a
good DJ does: not just "tracks that mix," but tracks in an *order with narrative
shape*.

**The journey arc (v1 default).** Grounded in how DJs actually build sets — a set
is chapters with deliberate tension and release, because the valleys are what make
the peaks hit:
1. **Intro / warm-up** — slow, melodic, ethereal, calmer. Sets mood, low intensity.
2. **Build** — rising energy and tension, tightening groove.
3. **Peak** — the high-vibe section; the emotional and energetic high point.
4. **Comedown / resolve** *(if appropriate)* — release the tension and end with a
   meaningful final statement, not a hard stop.

The arc is **not a monotonic climb**: it deliberately includes dips (release)
between peaks so the high points land. Each track's intensity (from §4.1's energy
summary) is matched to its position on the arc; harmonic moves (§4.3) are chosen
to *lift* into builds and *resolve* on the comedown.

**Algorithm.** Build the pairwise score matrix; find an ordering via
**greedy nearest-neighbour → 2-opt refinement**; the objective combines
**(a)** transition quality and **(b)** journey-arc adherence (intensity-vs-position
fit + tension/release shape). Each track used at most once (TSP-flavoured path,
not a cycle). Tracks that can't be placed without hurting the journey are reported
as stranded rather than forced in.

**One-shot intent.** The goal is that a single run on a folder yields a set good
enough to play with light human refinement — precision out of the gate, sharpened
over time. Quality of the *journey objective* is where that precision is won or
lost, so it is the primary tuning target (see §7's ear-tuning loop).

**Interface:** `sequence(tracks, arc_profile) -> list[OrderedTrack]`

> The arc profile is a parameter, which is what makes the future **Set Selector**
> (§8) — choosing *which* journey algorithm to run — a config swap, not a rewrite.

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
- **Set Selector (selectable journey algorithms)** — choose the sequencing
  objective that fits the moment, each a different weighting of the §4.4 objective:
  - **Just Mix** — optimise only for smooth transitions and things sounding good
    together; minimal narrative shape. Fast, safe, utilitarian.
  - **Experiential** — the full world-class headlining journey (think a Madison
    Square Garden closing set): strong narrative arc, deliberate tension/release,
    emotional peaks and resolution.
  - …and room for more (e.g. genre-specific or warm-up-only profiles).
  v1 ships a single default (the Experiential journey arc); because the arc/
  objective is already a parameter (§4.4), adding the selector is a config + CLI
  flag change, not a rewrite.
- **Real-time live mode** — listen to the currently playing audio (line-in or
  software capture), identify/locate the track, and surface "what mixes next"
  suggestions live from the pre-analysed library. Reuses the §4.3 scorer against
  the §4.2 store; the new work is low-latency audio capture + track identification.
  This is the "on-the-fly" half of the positioning, realised as a live engine
  rather than prep.

## 9. Environment & Tech Stack

### 9.1 Environment: conda (miniforge)

Selecta's environment is managed with **conda**, declared in a pinned
`environment.yml` at the repo root. Conda is the right tool here for reasons
specific to this project:

- **It owns the Python version.** Conda creates an isolated env with Python 3.12,
  fully independent of the machine's system Python 3.14. No `pyenv`, no global
  pollution — `conda activate selecta` and you're on the right interpreter.
- **It installs native (non-Python) dependencies too.** The audio stack isn't
  pure Python — librosa pulls compiled `numba`/`llvmlite`, and the project needs
  `ffmpeg`, `libsndfile`, and `rubberband`. Conda installs these *binaries into
  the env* from the **conda-forge** channel, so the project is self-contained and
  reproducible across machines (no "works on my mac" Homebrew drift).
- **It de-risks the hard libraries.** `madmom` and `essentia` (the future
  phrasing/DSP upgrades) are notoriously painful to `pip install` from source on
  macOS ARM, but ship as prebuilt conda-forge packages. Choosing conda now means
  those upgrades are a one-line `environment.yml` change later, not a build fight.
- **Reproducibility as infra.** `environment.yml` is checked into git and is the
  single source of truth for the environment; `conda env export` snapshots exact
  versions. This is the "robust, reproducible build" goal made concrete.

**Installer status:** conda 25.7.0 is already present (Homebrew miniconda, base at
`/opt/homebrew/Caskroom/miniconda/base`) with the fast **libmamba** solver already
default — no reinstall required. The one adjustment: the global config currently
lists only the `defaults` channel, but Selecta's audio stack (and the future
`madmom`/`essentia`) lives on **conda-forge**. Rather than change global config,
`environment.yml` pins `channels: [conda-forge]` for *this env only*, which also
sidesteps the `defaults`-channel commercial-use ToS. (A `defaults`-free,
conda-forge-first base like miniforge is a fine alternative but unnecessary here.)

**Day-one workflow:**
```bash
conda env create -f environment.yml   # build the env from the pinned spec
conda activate selecta                 # enter it
selecta analyze ~/Music/crate          # run
```

> Where it fits the pipeline: conda is the *foundation layer* under everything in
> §3 — every component runs inside the `selecta` conda env. It is the first thing
> the implementation plan stands up, before any feature code.

### 9.2 Libraries

- **Python 3.12** (via the conda env).
- **Audio/MIR:** `librosa`, `numpy`, `scipy`, `soundfile` (conda-forge). `madmom` /
  `essentia` reserved for the future phrasing/DSP upgrade.
- **Native deps (conda-forge):** `ffmpeg`, `libsndfile`, `rubberband`.
- **Storage:** SQLite via stdlib `sqlite3`.
- **Config:** TOML via stdlib `tomllib`.
- **CLI:** `argparse` or `click`.
- **Rekordbox XML:** stdlib `xml.etree.ElementTree` (no new dep). `pyrekordbox`
  only if/when the future direct-write backend is built.

## 10. References — DJ curation theory (informs §4.3/§4.4)

The journey arc and harmonic-move logic are grounded in established DJ practice:
- DJ sets as chapters with deliberate tension/release; valleys make peaks land;
  closing as a meaningful resolution, not just a fade
  ([djoid](https://www.djoid.io/articles/how-to-prepare-a-house-dj-set-in-chapters),
  [ZIPDJ](https://www.zipdj.com/blog/how-to-structure-a-dj-set),
  [DJs On Demand](https://www.djsondemand.co.uk/how-to-build-energy-in-a-dj-set/)).
- Camelot harmonic mixing — same key / ±1 / relative major↔minor mood shifts,
  clockwise = energy up
  ([Mixed In Key](https://mixedinkey.com/camelot-wheel/),
  [DJ.Studio](https://dj.studio/blog/camelot-wheel),
  [Audio Sorcerer](https://audiosorcerer.com/post/camelot-wheel)).
