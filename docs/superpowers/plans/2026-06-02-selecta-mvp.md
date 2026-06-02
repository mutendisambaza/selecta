# Selecta MVP — Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development (or Codex tickets via the orchestrator). Each task is TDD: write the failing test, implement minimally, run the test, commit. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A MIR pipeline that analyses a folder of electronic tracks, exports Rekordbox phrase cues, and sequences a journey-shaped DJ set — plan-only output.

**Architecture:** Two-phase pipeline. `analyze` extracts per-track MIR features (BPM, beat grid, Camelot key, energy curve, spectral profile, phrase map) into a SQLite store. Consumers read the store: a Cue Exporter (rekordbox.xml) and a Sequencer (journey arc) + Plan Renderer. Heuristic, explainable scoring; ML/embeddings/audio-render are out of scope (future).

**Tech Stack:** Python 3.12 (conda/conda-forge env), librosa + numpy + scipy + soundfile, stdlib sqlite3 / tomllib / xml.etree, pytest.

**Source spec:** `docs/superpowers/specs/2026-06-01-selecta-design.md`

---

## File Structure

```
selecta/
  environment.yml                  # conda env (python 3.12, conda-forge audio stack)
  pyproject.toml                   # package metadata + console_script `selecta`
  src/selecta/
    __init__.py
    cli.py                         # argparse: analyze / cues / sequence
    config.py                      # load config.toml (scoring weights, arc profiles)
    features/
      types.py                     # dataclasses: TrackFeatures, PhraseRegion, Beat
      extractor.py                 # extract(path) -> TrackFeatures
      tempo.py                     # bpm + beat grid + downbeats
      key.py                       # chroma -> Camelot key
      energy.py                    # energy curve + intensity scalar
      spectral.py                  # per-section frequency-band profile
      phrasing.py                  # downbeats -> phrase regions (intro/build/drop/...)
    store/
      db.py                        # FeatureStore (SQLite): upsert/get/all/needs_analysis
      hashing.py                   # audio content hash
    scoring/
      camelot.py                   # Camelot wheel adjacency + move classification
      scorer.py                    # score(a,b,arc_pos) -> TransitionScore
    sequencing/
      arc.py                       # journey arc profiles (intensity target by position)
      sequencer.py                 # sequence(tracks, arc) -> [OrderedTrack]
    render/
      plan.py                      # ordered set -> markdown + json
    export/
      rekordbox.py                 # export_cues(tracks, out_path)
  tests/
    fixtures/                      # tiny synthetic wavs + crafted TrackFeatures
    test_*.py
  config.toml                      # default weights + arc profiles
```

Each module has one responsibility; `extractor.py` orchestrates the `tempo/key/energy/spectral/phrasing` sub-extractors so each stays independently testable.

---

## Milestone M0 — Foundation  *(owner: orchestrator, done before agents)*

### Task 0.1: conda environment
**Files:** Create `environment.yml`

- [ ] Write `environment.yml`:
```yaml
name: selecta
channels: [conda-forge]
dependencies:
  - python=3.12
  - librosa
  - numpy
  - scipy
  - soundfile
  - ffmpeg
  - rubberband
  - pytest
  - pip
```
- [ ] Create env: `conda env create -f environment.yml`
- [ ] Verify: `conda run -n selecta python -c "import librosa, soundfile, numpy, scipy; print('ok', librosa.__version__)"` → prints `ok <version>`

### Task 0.2: package skeleton + smoke test
**Files:** Create `pyproject.toml`, `src/selecta/__init__.py`, `src/selecta/cli.py`, `tests/test_smoke.py`, `config.toml`

- [ ] `pyproject.toml` with `[project] name="selecta"`, `requires-python=">=3.12"`, console script `selecta = "selecta.cli:main"`, setuptools src layout.
- [ ] `cli.py` exposes `main()` with `argparse` subcommands `analyze`, `cues`, `sequence` (stubs that print "not implemented" and exit 0 for now).
- [ ] `tests/test_smoke.py`:
```python
def test_import_and_cli_help(capsys):
    from selecta import cli
    assert hasattr(cli, "main")
```
- [ ] `pip install -e .` inside the env, then `conda run -n selecta pytest tests/test_smoke.py -q` → PASS
- [ ] Commit: `chore: scaffold selecta package + conda env`

---

## Milestone M1 — Feature Extractor  *(depends: M0)*

Defines the data model and the MIR extraction. This is the foundation every consumer reads.

### Task 1.1: data types
**Files:** Create `src/selecta/features/types.py`, `tests/test_types.py`

- [ ] Define frozen dataclasses:
  - `Beat(time: float, is_downbeat: bool)`
  - `PhraseRegion(start: float, end: float, bars: int, label: str, confidence: float)`  — label ∈ {intro, build, drop, breakdown, outro, body}
  - `TrackFeatures(path, duration, bpm, bpm_confidence, beats: list[Beat], key_camelot: str, key_confidence: float, energy_curve: list[float], intensity: float, spectral: dict[str, dict[str,float]], phrases: list[PhraseRegion], title: str|None, artist: str|None)`
  - `to_dict()/from_dict()` for JSON/SQLite blob round-trip.
- [ ] Test: construct a `TrackFeatures`, round-trip through `to_dict`/`from_dict`, assert equality.
- [ ] Commit.

### Task 1.2: tempo + beat grid + downbeats
**Files:** Create `src/selecta/features/tempo.py`, `tests/test_tempo.py`, `tests/fixtures/click_120.wav`

- [ ] Generate fixture: a 120 BPM click track (8s) via numpy/soundfile in a `conftest.py` fixture or a committed wav.
- [ ] `detect_tempo(y, sr) -> (bpm: float, confidence: float, beats: list[Beat])`. Use `librosa.beat.beat_track`; infer downbeats as every 4th beat (4/4 assumption); confidence from beat-interval regularity.
- [ ] Test: on the 120 BPM fixture, `abs(bpm - 120) <= 2`; beats roughly every 0.5s; at least one downbeat flagged.
- [ ] Commit.

### Task 1.3: key detection → Camelot
**Files:** Create `src/selecta/features/key.py`, `tests/test_key.py`, `tests/fixtures/tone_Amin.wav`

- [ ] `detect_key(y, sr) -> (camelot: str, confidence: float)`. Chroma (`librosa.feature.chroma_cqt`) correlated with Krumhansl-Schmuckler major/minor profiles; map best key to Camelot (e.g. A minor → `8A`).
- [ ] Test: synth fixture in A minor (stacked A/C/E tones) → returns `8A` (or accept within ±1 Camelot, asserting confidence>0).
- [ ] Commit.

### Task 1.4: energy curve + intensity
**Files:** Create `src/selecta/features/energy.py`, `tests/test_energy.py`

- [ ] `energy_profile(y, sr, hop=...) -> (curve: list[float], intensity: float)`. RMS over frames, normalised 0–1; `intensity` = weighted mean (loudness + spectral flux proxy).
- [ ] Test: a signal that ramps in amplitude → curve is monotonically increasing (allow small noise); intensity in (0,1].
- [ ] Commit.

### Task 1.5: per-section spectral profile
**Files:** Create `src/selecta/features/spectral.py`, `tests/test_spectral.py`

- [ ] `band_profile(y, sr, regions) -> dict[label, dict[band,float]]` where bands = {low, low_mid, high_mid, high} energy fractions per region (intro/body/outro at minimum).
- [ ] Test: low-frequency sine → `low` band fraction dominant; high-frequency sine → `high` dominant.
- [ ] Commit.

### Task 1.6: phrase map
**Files:** Create `src/selecta/features/phrasing.py`, `tests/test_phrasing.py`

- [ ] `detect_phrases(beats, energy_curve, duration) -> list[PhraseRegion]`. Group downbeats into 8/16/32-bar phrases; label by position + energy (first phrase=intro, last=outro, max-energy plateau=drop, energy dip after drop=breakdown); set confidence lower when beat grid irregular (Tier-2 genres).
- [ ] Test: synthetic beats + an energy curve with a clear rise/peak/dip → at least intro, drop, outro labelled; phrase boundaries land on downbeats.
- [ ] Commit.

### Task 1.7: extractor orchestration
**Files:** Create `src/selecta/features/extractor.py`, `tests/test_extractor.py`

- [ ] `extract(path) -> TrackFeatures`: load audio (`soundfile`/`librosa.load`), read title/artist tags if present, call the sub-extractors, assemble `TrackFeatures`. Corrupt/undecodable file → raise `ExtractionError`.
- [ ] Test: on a committed short real-ish wav fixture, returns a populated `TrackFeatures` (bpm>0, valid camelot, non-empty phrases). Undecodable file raises `ExtractionError`.
- [ ] Commit.

---

## Milestone M2 — Feature Store  *(depends: M1.1)*

### Task 2.1: content hashing
**Files:** Create `src/selecta/store/hashing.py`, `tests/test_hashing.py`
- [ ] `audio_hash(path) -> str` (sha1 of file bytes, streamed). Test: stable across calls, differs for different content.
- [ ] Commit.

### Task 2.2: SQLite store
**Files:** Create `src/selecta/store/db.py`, `tests/test_store.py`
- [ ] `FeatureStore(db_path)` with `upsert(features)`, `get(path)->TrackFeatures|None`, `all()->list`, `needs_analysis(path)->bool` (true if absent or hash changed). Schema: scalar columns + JSON blobs for curves/phrases/spectral; key on path + hash.
- [ ] Test (use `tmp_path`): upsert a `TrackFeatures`, `get` round-trips it; `needs_analysis` false after upsert with same hash, true after content change; `all()` returns it.
- [ ] Commit.

---

## Milestone M3 — `analyze` CLI  *(depends: M1, M2)*

### Task 3.1: analyze command
**Files:** Modify `src/selecta/cli.py`, create `tests/test_cli_analyze.py`
- [ ] `selecta analyze <folder> [--db PATH]`: walk audio files, skip those where `needs_analysis` is false, extract + upsert, print a summary (n analysed / n skipped / n failed). Failures logged, not fatal.
- [ ] Test: point at `tests/fixtures/` with 1–2 wavs + a junk file; first run analyses, second run skips all; junk file counted as failed not crash.
- [ ] Commit.

---

## Milestone M4 — Cue Exporter + `cues` CLI  *(depends: M1, M2)*  — **first shippable win**

### Task 4.1: rekordbox xml export
**Files:** Create `src/selecta/export/rekordbox.py`, `tests/test_rekordbox.py`
- [ ] `export_cues(tracks: list[TrackFeatures], out_path: str)`. Emit `DJ_PLAYLISTS`→`COLLECTION`→`TRACK` with `Location` file URI + tags, and one `POSITION_MARK` per phrase boundary: memory cues (`Num=-1`) labelled by region; promote first-drop/main-breakdown/outro/intro to hot cues (`Num=0..7`, max 8). `Start` = absolute seconds. Low-confidence labels suffixed `?`.
- [ ] Test: build 1 `TrackFeatures` with known phrases, export, parse the XML back; assert required `POSITION_MARK` attrs (Name/Type/Start/Num) present, hot-cue `Num` ∈ 0..7, memory cues `Num=-1`, counts match phrases.
- [ ] Commit.

### Task 4.2: cues command
**Files:** Modify `src/selecta/cli.py`, create `tests/test_cli_cues.py`
- [ ] `selecta cues <folder> [--db PATH] [--out rekordbox.xml]`: load features from store (analyse new files first), call `export_cues`, write file. Test: end-to-end on fixtures produces a valid XML file.
- [ ] Commit.

---

## Milestone M5 — Scorer  *(depends: M1)*

### Task 5.1: Camelot wheel
**Files:** Create `src/selecta/scoring/camelot.py`, `tests/test_camelot.py`
- [ ] `relation(a: str, b: str) -> str` ∈ {same, adjacent, relative, energy_boost, distant} and `mood(a,b) -> str` ∈ {lift, darken, hold} (minor→major=lift, major→minor=darken). Encode the wheel (1–12, A=minor/B=major).
- [ ] Test: `8A,8A`→same/hold; `8A,9A`→adjacent; `8A,8B`→relative/lift; `8B,8A`→relative/darken; `8A,2A`→distant.
- [ ] Commit.

### Task 5.2: pairwise scorer
**Files:** Create `src/selecta/scoring/scorer.py`, `src/selecta/config.py`, `tests/test_scorer.py`
- [ ] `score(a, b, arc_pos: float, weights) -> TransitionScore` with per-dimension breakdown {harmonic, tempo, energy, spectral, phrase} and total 0–100. Direction-aware harmonic (uses `mood` vs what arc wants at `arc_pos`). Tempo: smaller %stretch better, half/double compatible. Spectral: complementary outro/intro bands score higher.
- [ ] Test: identical-key same-BPM tracks score high; distant-key large-BPM-gap score low; breakdown keys all present; returns a structured object not a bare float.
- [ ] Commit.

---

## Milestone M6 — Sequencer  *(depends: M5)*

### Task 6.1: arc profiles
**Files:** Create `src/selecta/sequencing/arc.py`, `tests/test_arc.py`
- [ ] `arc_target(profile: str, pos: float) -> float`: intensity target (0–1) at normalised position. `experiential` profile = ethereal-low → build → peak (~0.7 through) → resolve, with deliberate mid dips (not monotonic). Loadable from `config.toml`.
- [ ] Test: experiential profile is low at pos 0, high near peak, lower at 1.0; includes at least one local dip.
- [ ] Commit.

### Task 6.2: journey sequencer
**Files:** Create `src/selecta/sequencing/sequencer.py`, `tests/test_sequencer.py`
- [ ] `sequence(tracks, profile, weights) -> list[OrderedTrack]`: pairwise score matrix → greedy nearest-neighbour seed → 2-opt refinement; objective = transition quality + arc adherence (track intensity vs `arc_target`). Each track once; unplaceable tracks returned as `stranded`.
- [ ] Test: synthetic library with intensities {0.1..0.9} and compatible keys → output orders intensities into the arc shape (low→high→down), each track used once.
- [ ] Commit.

---

## Milestone M7 — Plan Renderer + `sequence` CLI  *(depends: M6, M1)*

### Task 7.1: plan renderer
**Files:** Create `src/selecta/render/plan.py`, `tests/test_plan.py`
- [ ] `render(ordered, fmt) -> str` for `markdown` and `json`. Per-transition line: cue bar/time (align A outro phrase ↔ B intro phrase on downbeats), both Camelot keys, %BPM nudge, bass-swap bar. Mark low-confidence transitions.
- [ ] Test (golden): given a fixed 3-track ordered set, markdown contains each track + transition instructions; json parses and has per-transition fields.
- [ ] Commit.

### Task 7.2: sequence command
**Files:** Modify `src/selecta/cli.py`, create `tests/test_cli_sequence.py`
- [ ] `selecta sequence <folder> [--arc experiential] [--out PATH]`: load/analyse → sequence → render md+json. Test: end-to-end on fixtures yields a markdown plan and json file.
- [ ] Commit.

---

## Self-Review (orchestrator checklist)
- **Spec coverage:** Extractor(M1)=§4.1; Store(M2)=§4.2; Scorer(M5)=§4.3; Sequencer(M6)=§4.4; Renderer(M7)=§4.5; Cues(M4)=§4.6; CLI(M3/4.2/7.2)=§4.7; conda(M0)=§9. Futures (ML/embeddings/audio-render/live/set-selector) intentionally absent.
- **Dependency order:** M0→M1→{M2→M3, M4 after M2, M5→M6→M7}. M4 and the M5–M7 chain are parallelisable once M1+M2 land.
- **TDD:** every task = failing test → minimal impl → passing test → commit.
