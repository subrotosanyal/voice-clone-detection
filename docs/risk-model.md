# The risk model — transparency, explainability, reproducibility

## The formula

```
fused_raw = sum(weight_effective[i] * score[i])   over every non-abstaining detector i
```

where `weight_effective` renormalises the *configured* weights of only the
detectors that actually produced a score, so they always sum to 1 — an
abstained signal (no enrollment on file, near-silent window, no call
metadata) is dropped from the sum entirely rather than silently counted
as a zero and dragging the score down. See
`tests/unit/test_fusion_weighted_sum.py::test_abstained_signal_is_renormalised_not_zeroed`
for the exact behaviour, asserted.

The raw score is then EMA-smoothed against the session's previous
smoothed score (`fusion.smoothing_alpha` in config — higher reacts faster,
lower is steadier), and banded:

| Band | Range (configurable) | Meaning |
|---|---|---|
| `low` | 0–34 | Consistent with natural speech. No interruption. |
| `elevated` | 35–69 | Some indicators present. Quiet advisory only. |
| `high` | 70–100 | Strong indicators. Block the pending sensitive action. |

All of this lives in `config/risk_formula.yaml` and
`app/adapters/fusion/weighted_sum.py`.

## Transparency & explainability

Every `FusedScore` returned by the API (and logged) carries a `components`
list — one entry per detector, with:

- `raw_score` — what the detector itself returned (or `null` if it abstained)
- `weight_configured` / `weight_effective` — before and after renormalisation
- `contribution` — exactly what this detector added to the final number
- `detail` — the concrete feature values the score came from (e.g.
  `spectral_flatness: 0.71`), plus the actual implementation's
  `detector_name`/`detector_version`, so a score is traceable to real code,
  not just a config-level label
- `abstained` / `abstain_reason` — when and why a signal didn't vote

If you can't answer "why is this score 62 and not 40" from the JSON alone,
that's a bug in a detector's `detail`, not something to fix by reading
source code.

## Reproducibility

`fuse()` is a pure function of `(results, previous_smoothed_score, config)`
— nothing else. Given the same three inputs, it always returns the same
`FusedScore`. That in turn means:

- Re-running `POST /v1/score/file` with the same audio bytes and the same
  `context` always produces the same score (asserted in
  `tests/integration/test_api_score_file.py::test_repeated_calls_with_same_input_are_reproducible`).
- `formula_version` (in `config/risk_formula.yaml`) is stamped onto every
  `FusedScore` and every log line. **Bump it whenever a config change would
  change past scores if replayed** — that's what lets you look at an old
  logged score and know exactly which formula produced it.
- Detectors are required (by the `DetectorPort` docstring) to be
  deterministic for the same `(window, context, detector_version)`. A
  detector that calls an external model must pin and record that model's
  version in `detail` for the same reason.

## The acoustic detector: a real model now, with real limits

`acoustic_aasist.py` wraps a genuine pretrained countermeasure (AASIST,
ASVspoof2019 LA) rather than a hand-rolled heuristic — see
`docs/architecture.md` for what it does and doesn't prove. In short: real
English speech vs. real English TTS/VC attacks, nothing about Indian
languages or about non-speech audio. Attribution and licence:

- Source: https://github.com/clovaai/aasist, commit `a04c9863`
- Licence: MIT, Copyright (c) 2021-present NAVER Corp. — full text in
  `app/adapters/detectors/vendor/AASIST_LICENSE`
- Checkpoint fetched and checksum-verified by
  `vendor/fetch_checkpoint.py`, not committed to git

## The prosodic detector: Parselmouth (Praat), with real limits

`prosody_parselmouth.py` (the default `prosodic` class as of
`formula_version: "2026.09.4"`) computes jitter, shimmer, and harmonics-to-
noise ratio via **Parselmouth** — the official Python binding for **Praat**,
the long-standing reference tool in clinical voice-quality research.
Attribution and licence:

- Source: https://github.com/YannickJadoul/Parselmouth
- Licence: GPLv3
- These are real, validated acoustic-phonetic measurements — a
  meaningfully more validated *front end* than the previous autocorrelation
  pitch-variance heuristic (`prosody_pitch_variance.py`, still available
  via a one-line config swap for a Praat-free run).

**What's still a placeholder**: the risk-score MAPPING built on top of
those Praat measurements (`_JITTER_FLOOR`/`_SHIMMER_FLOOR`/`_HNR_CEILING`
in the module) is an unvalidated heuristic — "an unnaturally smooth voice
is synthetic-suspicious" — not a trained classifier, and not yet tuned
against a labeled genuine-vs-synthetic corpus. The lightweight acoustic
fallback (`acoustic_spectral_flatness.py`) has the same class of caveat.
Treat any score that includes either as partly proof-of-pipeline, not a
fully validated fraud signal — the acoustic-via-AASIST component remains
the one part of today's score backed by a real, published, trained
*classifier* (as opposed to a validated feature extractor with a heuristic
threshold on top).

## Voiceprint consistency: real ECAPA-TDNN comparison, with real limits

`voiceprint_consistency.py` (the default `consistency_class`) compares a
live call's speaker embedding against one enrolled earlier for the claimed
identity — see `docs/architecture.md`, "Voiceprint consistency" for how.
Attribution and licence:

- Model: `speechbrain/spkrec-ecapa-voxceleb` (SpeechBrain, ECAPA-TDNN
  architecture, trained on VoxCeleb1+2)
- Licence: Apache-2.0
- Confirmed ungated on HuggingFace (`gated: false`) — no auth token needed
  to build or run this image, unlike pyannote.audio's pretrained pipelines
  (all gated) which were rejected for exactly that reason — see
  `app/adapters/embeddings/ecapa_embedding.py`'s docstring for the full
  dependency-selection reasoning (Resemblyzer was also considered and
  rejected, for an unmaintained `webrtcvad` dependency).

**What's still a placeholder**: the cosine-similarity->risk mapping
(`_MATCH_SIMILARITY`/`_NO_MATCH_SIMILARITY` in the module) is a placeholder
anchor pair, not a threshold picked off an EER curve on labeled same/
different-speaker pairs run through this exact preprocessing pipeline.
Speaker embedding comparison is a real, well-established technique (the
same approach the SASV Challenge 2022 baseline uses); the specific numeric
thresholds here are not yet calibrated.

## Diarization: real clustering, with real limits

`embedding_cluster_diarizer.py` reuses the same ECAPA-TDNN embedding
extractor to answer "roughly how many people spoke, and which stretches
were whose" for an uploaded recording — see `docs/architecture.md`,
"Diarization" for how. Real agglomerative clustering
(`scipy.cluster.hierarchy`, cosine distance), not a lookup table. **What's
still a placeholder**: fixed-length segmentation (not a proper voice-
activity/change-point front end) can miss a speaker change mid-segment,
and the clustering distance threshold is a placeholder pending calibration
on labeled multi-speaker audio. Good enough for "roughly how many voices,
roughly which stretches" — not turn-by-turn transcription-grade diarization.

## Transcription and urgency-language detection

`whisper_transcriber.py` transcribes an uploaded call once via OpenAI's
Whisper. Attribution and licence:

- Model: `openai/whisper-small` (also available: tiny/base/medium/large)
- Licence: MIT
- Confirmed ungated on HuggingFace — no auth token needed

**Why "small", specifically**: verified by hand, not assumed. The
smaller "base" model mis-transcribed real Hindi speech into Urdu script —
a genuine failure mode, not a hypothetical one (the two languages sound
alike but are written differently, and "base" isn't reliable enough to
keep them straight). "small", tested on the same audio, got both the
language and the script right.

**Why keyword matching, not a sentiment-analysis model**: the signal this
project actually needs isn't generic positive/negative sentiment — it's
specific, fraud-relevant language ("transfer the money now", "don't tell
anyone"). `urgency_language.py`'s keyword lists keep every match traceable
to an actual word in the actual transcript, shown in
`ContextualRulesDetector`'s `detail` — consistent with the contextual
mode's whole design goal (§03: "legible to a non-technical reader",
"no ML or enrollment infra"). A black-box sentiment score would trade that
transparency away for a kind of signal this project doesn't actually need.

**What's still a placeholder**: the keyword lists (English/Hindi/Marathi)
were drafted for thematic coverage, not validated against real fraud-call
transcripts — same class of caveat as the prosodic/voiceprint detectors'
placeholder thresholds elsewhere in this document. Real ASR, real keyword
matches; the specific word lists are a starting point.

## The pluggable third signal, pros/cons

| Mode | Pros | Cons |
|---|---|---|
| `contextual` | No ML/enrollment infra; zero privacy burden; legible to a non-technical reader; works from the first call | Weak alone — an attacker who knows the target's number and calls in business hours evades it entirely |
| `consistency` | Checks identity directly (real ECAPA-TDNN comparison, see above); harder to defeat even for a sophisticated attacker | Needs an enrollment flow (built — `POST /v1/enroll`), consent record, embedding store, TTL (retention/consent policy still manual); cold-start for anyone not enrolled |
| `auto` (default) | Consistency when available, contextual fallback otherwise — no silent zero | Slightly more moving parts to reason about than a fixed mode |

Switch modes in `config/risk_formula.yaml` → `detectors[third_signal].params.mode`
— no code change.
