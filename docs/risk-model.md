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

**`detail.explanation`** — every non-abstaining detector also writes one
plain-language sentence into its own `detail`, shown directly under that
signal's bar in the dashboard (not just in "view raw JSON"). It's built
from the same numbers already in `detail`, nothing extra: contextual names
which rules fired and which words triggered them; prosodic names whichever
of jitter/shimmer/HNR actually pushed the score, with the measured value
next to its reference; voiceprint consistency states the measured cosine
similarity against both anchors. The acoustic detector (AASIST) is the
deliberate exception: it's a trained neural classifier, not a rule engine,
so its explanation states its confidence honestly ("the model assigned
X% probability this is synthetic, based on its internal representation")
rather than inventing a feature-level reason it doesn't actually have —
see `acoustic_aasist.py`'s `_build_explanation()` docstring.

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

**Hindi recalibration (wired into production 2026-09-08)**: the `acoustic`
detector's `finetuned_out_layer_path` param (`config/risk_formula.yaml`)
swaps in a small (~3KB), in-house-produced recalibration of just AASIST's
final classification layer — the base model, architecture, and training
data are all unchanged; only `out_layer`'s weights differ. Produced by
`eval/indian_language`'s fine-tuning pipeline (see that directory's
README.md for the full reproducible process, including two real bugs
found and fixed along the way: a class-imbalance issue that let an
earlier attempt collapse toward always predicting "bonafide", and a
BatchNorm/Dropout statistics-drift bug that made an even earlier attempt
non-reproducible between training time and a fresh reload). Honest
numbers: EER on the in-house eval corpus dropped 17.75% -> 3.0%; more
meaningfully, false-positive rate on an independent, more varied real
Hindi speech sample (never used in training) dropped 75% -> 20%. Every
score's `detail.hindi_finetuned_out_layer` flag says whether this
recalibration was active, so nothing here is hidden. Not "solved" —
20% is still a real, material false-positive rate, just far better than
before.

## The prosodic detector: Parselmouth (Praat), with real limits

`prosody_parselmouth.py` (the default `prosodic` class as of
`formula_version: "2026.09.7"`) computes jitter, shimmer, and harmonics-to-
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
(`scipy.cluster.hierarchy`, cosine distance), not a lookup table.

**Recalibrated 2026-09-08, twice, after a real over-counting bug**: one
person's natural voice variation (breath, background noise, prosody
drift) was splitting into multiple "speakers" (a real report: over 100
phantom speakers for one call). The first same-day fix moved the
clustering cutoff the WRONG direction (0.35→0.28 — lower is *stricter*,
which makes over-counting worse, not better) and was confirmed broken
immediately by that same report. Corrected: the cutoff now sits at 0.4
(higher than even the original), and segments were lengthened
1500ms→2500ms (more audio per embedding, independently useful regardless
of the cutoff value). A `max_speakers` cap (default 8) was also added as
a real safety net — no matter what the distance cutoff turns out to be
wrong about on some future recording, this diarizer will never again
report more speakers than that. See `embedding_cluster_diarizer.py`'s
CALIBRATION HISTORY docstring note for the full account, including the
reasoning error, kept rather than deleted so it doesn't happen again.

**Segmentation redesigned 2026-09-08 (same day, after the above): pause-aware,
not fixed-length.** Investigating whether simply lengthening segments would
help surfaced a second, distinct over-counting mechanism: a segment that
straddles a real speaker change produces an ECAPA-TDNN embedding that
resembles NEITHER speaker (measured by hand on synthetic splices: cosine
similarity 0.04-0.34 against either pure voice, far below the normal
cross-speaker baseline of ~0.67-0.70) — not a "blend", a phantom cluster of
its own. `pause_segmentation.py` now cuts segment boundaries at real
acoustic pauses (Praat's own silence detector — no new dependency,
Parselmouth is already required for the prosodic detector) instead of a
fixed clock, so a segment no longer straddles a detected speaker turn.
`max_segment_ms` (still 2500 by default) is now only a cap for sub-slicing
long uninterrupted stretches, not the primary segmentation unit. **Honest
limit, verified by hand**: a ~150ms gap between speakers is reliably
detected; a genuine zero-gap turn-take (immediate back-to-back speech, or
overlap) leaves no acoustic silence for any pause-based method to find, no
matter how it's tuned — `max_segment_ms` bounds the damage from a missed
zero-gap turn to at most one chunk, same as the old fixed-grid behaviour,
rather than eliminating the case. See `pause_segmentation.py`'s own HONESTY
NOTE for the measurements.

**What's still a placeholder**: the distance cutoff remains an unvalidated
guess (no real labeled multi-speaker corpus exists in this repo — see
`eval/indian_language/README.md`'s own note on why that data doesn't
exist either), and a genuine zero-gap turn-take is still undetectable by
this or any pause-based method (see above). The `max_speakers` cap is the
part of this you can actually trust regardless. Good enough for "roughly
how many voices, roughly which stretches" — not turn-by-turn
transcription-grade diarization.

**Live path has its own, separate tracker (added 2026-09-09)**: this
diarizer needs a COMPLETE recording before it can cluster, so it does
NOT run on the live WebSocket path. `app/pipeline/live_diarization.py`
(`LiveSpeakerTracker`) answers the same question incrementally instead —
same ECAPA-TDNN embeddings, a simpler nearest-centroid-with-a-threshold
algorithm. It does not feed the risk formula at all (purely
informational — `live_speaker` on the API response), so it's not part
of "the formula" this document otherwise covers; see
`docs/architecture.md`, "Live diarization" for the full account,
including its own separate calibration caveat.

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

**Reproducibility fix, 2026-09-08**: a real intermittent bug was found and
fixed — Whisper's `transcribe()` defaults to a temperature FALLBACK tuple,
not a single fixed value, and on non-speech audio (silence, a pure tone —
exactly this project's synthetic test fixtures) the initial greedy pass can
fail Whisper's own quality gates and fall back to sampling, which draws
from PyTorch's global, unseeded RNG. Verified by hand: this made
transcription genuinely non-deterministic call to call on identical audio
within one process, sometimes returning an empty transcript and sometimes
a hallucinated one ("MMMMMMMMMMMMMM", observed directly) — which then fed
the intent classifier and changed the final fused score run to run. Fixed
by pinning `temperature=0.0`, which forces pure greedy decoding — see
`whisper_transcriber.py`'s own REPRODUCIBILITY FIX docstring note for the
full account and the trade-off this accepts (Whisper's one real recovery
path, retrying at higher temperature on hard audio, is now disabled).

**Authority-claim detection and the combined-pressure rule (added
2026-09-08)**: `urgency_language.py` also scans for a third category —
the caller asserting they're a bank, police, or government official
("this is your bank", "cyber crime cell", "arrest warrant"). Alone this
is weak evidence (a real bank call also opens that way); its real weight
comes from `ContextualRulesDetector`'s new `combined_authority_financial_pressure`
rule, which fires only when an authority claim AND a financial request
appear in the same call — the classic fraud script ("this is your bank,
your account has been compromised, transfer your funds now") scored as
its own explicit, named rule rather than silently summed from the two
components. Same starting-point caveat as the other keyword lists above.

## Intent detection (zero-shot classifier) — built, verified, active with a known caveat

`app/adapters/intent/zero_shot_intent_classifier.py` classifies a
transcript against five fixed candidate labels (`"requesting a money
transfer or payment"`, `"requesting an OTP or PIN"`, `"impersonating a
bank or government official"`, `"creating urgency or time pressure"`,
`"ordinary conversation"`) using zero-shot NLI classification — meant to
generalise beyond exact keyword matches, while staying explainable: every
score traces back to "the model judged this X% consistent with candidate
Y", for every candidate, not just the winner.

**Model — verified before adoption**:
- Source: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
- Licence: MIT — an explicit tag on the model card, confirmed via the HF
  API. No ambiguity, unlike the sentiment-analysis candidate discussed
  (but not adopted) elsewhere in this project.
- Confirmed ungated on HuggingFace
- Real Hindi coverage: fine-tuned on the XNLI dataset, which includes
  Hindi among its 15 languages. Marathi is NOT in XNLI — any Marathi
  result would be unvalidated cross-lingual transfer from the base
  model's 100-language pretraining, same caveat class as the cardiffnlp
  sentiment model discussed for the separate, still-unbuilt sentiment
  feature.

**The known calibration problem, surfaced before enabling this, not
after**: three real transcripts were run through it by hand — a clear
fraud script, a clear impersonation script, and two completely ordinary
sentences ("are we still on for dinner tonight?", "the weather has been
really nice this week"). The fraud/impersonation examples classified
sensibly. Both ordinary sentences did NOT: "dinner tonight" scored just
4.5% for "ordinary conversation" against 38.9% for "creating urgency or
time pressure" — a severe false positive on completely benign text. This
was tested three ways (default settings, `multi_label=True`, and a
phone-call-specific `hypothesis_template`) and the miscalibration
persisted in all three — see `zero_shot_intent_classifier.py`'s own
HONESTY NOTE for the full numbers.

**Enabled anyway, at explicit user instruction, with two real
mitigations**: a deliberately low fusion weight (`0.15` in
`config/risk_formula.yaml`, auto-renormalised against
acoustic/prosodic/third_signal — see "The formula" above — so this alone
can't push a genuine bonafide call into High), and full UI transparency:
the dashboard shows every candidate label's own score for the "Intent"
component, not just the winner, specifically so a false positive here is
visible and inspectable rather than hidden inside one opaque number. This
does not fix the miscalibration — it means an operator sees "creating
urgency or time pressure: 39%, ordinary conversation: 4%" laid out in
full and can judge for themselves, the same trust model this project's
other honestly-flagged heuristic thresholds (prosodic, voiceprint
consistency) already rely on. Real fixes not done yet: several distinct
"ordinary" hypotheses instead of one, a real held-out evaluation set, or
a different base model.

**What's real and working**: the port (`app/ports/intent_classifier.py`),
the adapter above, the detector (`app/adapters/detectors/intent_risk.py`,
which reads a pre-computed classification out of `context` rather than
calling the model per-window — the same "compute once per call" pattern
transcription uses, for the same reason: an NLI forward pass is
expensive), and — as of 2026-09-08 — BOTH call paths: `Engine.
score_call()`/`http_router.py` populate `context["intent_label"]` once
per whole-call upload; `app/pipeline/live_transcription.py`/`ws_router.py`
populate the same keys periodically in the background on the live
WebSocket path (see "Live transcription" in `docs/architecture.md`),
lagging real speech by design rather than blocking the per-window score.
To disable it again, comment out the `intent_classification:` section
and the `intent` entry under `detectors:` in `config/risk_formula.yaml` —
the detector then simply abstains on every window, on either path.

## Watermark check (Perth) — built, verified, active, narrow by design

`app/adapters/detectors/perth_watermark.py` checks for Resemble AI's
Perth neural watermark — MIT licensed, confirmed via the GitHub API's own
license detection (the PyPI package's own metadata leaves the license
field blank, so this was verified independently rather than trusted at
face value). Chatterbox and other Perth-integrated voice-cloning tools
embed this specific fingerprint in every clip they generate; unlike
AASIST's general "does this sound synthetic" judgment, this is closer to
checking for a known signature.

**Verified by hand, not assumed**:
- 10 real genuine speech samples (2 macOS `say`-synthesized English
  clips, 8 real Hindi speakers) read near-zero: 9/10 below 0.04, one
  outlier at 0.36 — still well below any reasonable 0.5 threshold.
- A DIFFERENT, non-Perth TTS system (Coqui XTTS-v2, already used in
  `eval/indian_language/`) also reads near-zero on 5 synthetic samples —
  confirms this is specific to Perth's watermark, not a generic "sounds
  synthetic" trigger that would duplicate AASIST.
- A clip actually run through Perth's own `apply_watermark()` reads 1.0.
- Inference is fast (~7ms per 2s window on CPU) — no real-time concern.

**Known false-positive, found and documented, not hidden**: a pure sine
tone (this project's own non-speech test fixture) reads 0.92. This
detector's real-world scope is genuine/cloned speech, same as every
other detector here — a synthetic tone was never a realistic phone-call
input to begin with.

**Narrow scope, honestly stated**: this only catches watermark-compliant
generators. It says nothing about tools that don't watermark their
output, or about audio where the watermark was deliberately stripped —
a low score here means "this specific fingerprint wasn't found", never
"this audio is genuine". Weighted low (0.15, same as `intent`, auto-
renormalised) for exactly that reason; it complements AASIST's
general-purpose judgment rather than replacing it.

## The pluggable third signal, pros/cons

| Mode | Pros | Cons |
|---|---|---|
| `contextual` | No ML/enrollment infra; zero privacy burden; legible to a non-technical reader; works from the first call | Weak alone — an attacker who knows the target's number and calls in business hours evades it entirely |
| `consistency` | Checks identity directly (real ECAPA-TDNN comparison, see above); harder to defeat even for a sophisticated attacker | Needs an enrollment flow (built — `POST /v1/enroll`), consent record, embedding store, TTL (retention/consent policy still manual); cold-start for anyone not enrolled |
| `auto` (default) | Consistency when available, contextual fallback otherwise — no silent zero | Slightly more moving parts to reason about than a fixed mode |

Switch modes in `config/risk_formula.yaml` → `detectors[third_signal].params.mode`
— no code change.
