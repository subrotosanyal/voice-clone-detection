# AASIST / Prosodic recalibration — data expansion & fine-tuning plan

Working doc for the current push to grow Hindi training data beyond the
existing 200 genuine + 40 synthetic examples, and use it to improve
AASIST and prosodic detector performance. Updated as work progresses —
see README.md's "Datasets" table for the license verification behind
every source below.

## Source verification (done)

Checked all 8 URLs the user provided, plus pivot attempts once the first
one turned out unusable:

| Source | Verdict | Why |
|---|---|---|
| `agarwalayushi/hinglish` | **Rejected** | 815K-clip aggregator, CC BY 4.0 top-level tag, but compiled from 14 sub-sources — 2 have no independently verified license (`krishan23/indian_english`, `ar17to/orpheus_tts...`, ~25K clips). Its `source` field is a per-SPEAKER id, not a per-sub-dataset label, so those rows can't be filtered out. Not used wholesale. |
| `ai4bharat/Mann-ki-Baat` | **Blocked** | Gated on HF (`gated: auto`), even though CC BY 4.0. No ungated mirror found. |
| `krishan23/indian_english` | **Blocked** | No license declared anywhere. |
| `ai4bharat/Kathbath` (direct HF) | **Blocked directly, usable via pivot** | Gated on HF, but same CC BY 4.0 data is mirrored ungated via AI4Bharat's own IndicSUPERB GitHub repo (E2E Networks direct download, no auth). |
| `SPRINGLab/IndicTTS-Hindi` | **Already in use** | Existing 200-example genuine set (`fetch_genuine_corpus.py`). |
| `ar17to/orpheus_tts_english_indian_multispeaker` | **Blocked** | No license declared. |
| `github.com/AI4Bharat/indicSUPERB` | **Confirmed viable — in progress** | Direct-download mirror of Kathbath (CC BY 4.0). Fetch script: `scripts/fetch_kathbath_hindi.py`, running now. |
| `github.com/mrinmoy-iitg/Movie-MUSNOMIX` | **Confirmed viable — done** | Real CC0 annotations; underlying movie audio is academic/non-commercial-only, and the user explicitly confirmed (2026-09-09) this project qualifies. Fetch script: `scripts/fetch_movie_musnomix.py`. |

Also checked as pivots after the Hinglish rejection, both blocked:
- `ai4bharat/Spoken-Tutorial` (NPTEL) — gated on HF, no ungated mirror found.
- `mozilla-foundation/common_voice_17_0` — deprecated on HF as of Oct 2025;
  Mozilla now distributes exclusively via "Mozilla Data Collective", a
  platform not yet investigated. Dropped from the source list rather than
  chased further, given Kathbath alone is a substantial (150+ hour)
  properly-licensed addition.

**Net new sources fetched in Phase 1: Kathbath (500 Hindi utterances, CC BY
4.0, via the ungated IndicSUPERB/E2E-Networks mirror) + Movie-MUSNOMIX
(203 Hindi movie-dialogue segments, CC0 annotations / non-commercial-use
underlying audio, user-confirmed)**, plus the local 100-speaker set for
Phase 2.

## Phase 1 — Data pipeline setup (done)

- [x] Organize the local 100-speaker `TrainingAudio`/`TestingAudio` into
      the eval pipeline (`data/local_speakers/{train,test}/`)
- [x] Write and run `scripts/fetch_kathbath_hindi.py` — **500/500 Hindi
      utterances fetched** to `data/genuine/hi_kathbath/` (real bug found
      and fixed mid-run: transcript keys retained the `.m4a` suffix,
      causing every lookup to silently miss — fixed, and the manifest's
      "text" fields were patched in place afterward without redoing the
      3.6GB download/extraction; all 500 entries now carry real
      transcript text)
- [x] Write and run `scripts/fetch_movie_musnomix.py` — **203/203 Hindi
      speech-class segments fetched** to `data/genuine/hi_movie_musnomix/`
      (real Bollywood dialogue, 4 films spanning 1966-2011 — a genuinely
      different acoustic domain from both IndicTTS-Hindi's studio
      recordings and Kathbath's crowd-sourced read speech)

## Phase 2 — Matched spoof/deepfake data for the 100 local speakers (mostly done)

The local `TrainingAudio`/`TestingAudio` set is genuine-only — no spoof
counterpart, so it can't be used for detector training as-is.

- [x] Generate cloned/spoof audio per local speaker via XTTS-v2, using
      each speaker's own longest training clip as reference — `scripts/
      generate_local_speaker_clones.py`, **300/300 clips done** (100
      speakers x 3 sentences each; resumable, so re-running with more
      sentences per speaker later just adds to this, doesn't redo it)
- [x] Manifest pairing — no separate join file needed: `data/
      synthetic_local_speakers/manifest.json`'s `speaker_id` field
      matches the genuine side's folder name (`data/local_speakers/
      train/<speaker_id>/`) directly
- [ ] Chatterbox as a SECOND, held-out cloning system for these same 100
      speakers (same held-out-generalization discipline already used for
      the smaller reference-speaker set — see README.md's "Held-out
      generalisation check") — not done yet, optional before Phase 4's
      generalization check, since Chatterbox held-out data already
      exists from the smaller set

## Phase 3 — Prosodic threshold calibration (done — real, concerning finding)

- [x] Extract jitter/shimmer/HNR across all labeled genuine+spoof data
      (1423 genuine train / 340 spoof train / 100 genuine held-out / 40
      spoof held-out) — `scripts/calibrate_prosodic_thresholds.py`,
      verbose per-file progress, results in `results/
      prosodic_calibration.json`
- [x] Calibrated — **the currently-deployed thresholds
      (`_JITTER_FLOOR`/`_SHIMMER_FLOOR`/`_HNR_CEILING` in
      `prosody_parselmouth.py`) score exactly 50% balanced accuracy —
      chance level** (100% genuine accuracy, 0% spoof accuracy, on BOTH
      train and held-out data: they never flag anything as spoof against
      real data). Root cause isn't threshold placement — genuine vs.
      spoof jitter/shimmer/HNR distributions are nearly IDENTICAL in this
      corpus (e.g. mean jitter 0.0203 genuine vs. 0.0226 spoof), so a
      percentile-based single-feature recalibration barely moves the
      needle (still ~50% balanced accuracy). A logistic regression
      COMBINING all three features (standardized, `class_weight=
      "balanced"`) does meaningfully better: **82.5% balanced accuracy
      on genuinely held-out data** (70% genuine / 95% spoof; held-out =
      Chatterbox-cloned spoof + local speakers' own held-out test
      utterances, neither used in fitting) — a real, substantial
      improvement over chance, though still an open question whether
      3 scalar features are enough for a production detector.
- [x] **Verbose progress reporting** — per-file extraction progress
      (every 25 files), running abstain counts, full distribution/
      performance summary, all printed live during the run

**Not yet decided (Phase 5's call, not made unilaterally here)**: whether
to actually replace `ParselmouthProsodyDetector`'s hand-tuned formula
with this trained classifier in production — that's an architecture
change (serialized model behind the same `DetectorPort`, not just a
config constant), not just a threshold edit to risk_formula.yaml.

## Phase 4 — AASIST fine-tuning

- [ ] Extend `scripts/fine_tune_aasist.py` with the larger, more diverse
      dataset (existing IndicTTS-Hindi + Kathbath genuine, existing
      XTTS-v2/Chatterbox synthetic + new local-speaker-cloned synthetic)
- [ ] **Verbose progress reporting required** — the script already prints
      dataset-build progress, class balance, per-50-step and per-epoch
      metrics, best-epoch selection; extend this, don't rebuild it
- [ ] Held-out generalization check: never test on a synthesis system
      used in training (same discipline as the existing XTTS-v2/held-out
      Chatterbox split)

## Phase 5 — Reporting & deployment decision

- [ ] Honest before/after numbers for both AASIST and Prosodic (same
      format as README.md's existing before/after tables)
- [ ] Decide whether to update `config/risk_formula.yaml` based on the
      real numbers, not before
