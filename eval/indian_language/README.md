# Indian-language dataset & held-out evaluation (blueprint §04)

This is the offline pipeline that produces the "Indian-language gap" numbers
referenced throughout `docs/blueprint.html` (§00, §04, §08). **As of
2026-09-08 it DOES touch the live-call runtime**: `results/
aasist_out_layer_finetuned.pth` (the fine-tuned output layer this pipeline
produces) is copied to `services/live-call-api/app/adapters/detectors/
vendor/checkpoints/AASIST_hindi_finetuned_out_layer.pth` (a small, ~3KB,
in-house artifact — committed directly to git, not fetched, no license
concern the way a third-party download would have) and wired in via
`config/risk_formula.yaml`'s `finetuned_out_layer_path` param — see
`docs/risk-model.md`'s "Hindi recalibration" note. Re-running this
pipeline's scripts and re-copying the result is how you refresh that
production artifact; nothing here happens automatically at deploy time.
See `docs/architecture.md`'s second diagram ("Build-time — the
Indian-language dataset & evaluation pipeline") for how this fits the
overall project.

## Scope: English and Hindi only, for now

The blueprint's original brief named Hindi, Marathi and Malvi. The team has
deliberately narrowed active scope to **English (the existing AASIST
baseline) and Hindi** — Marathi and Malvi are no longer required
deliverables. Reasoning: Malvi has no digital resources of any kind and
would need a native-speaker volunteer to record from scratch (see
`CONSENT_FORM.md`, still ready for exactly that if one ever joins), and no
ungated open TTS/cloning system covering Marathi was found, so neither
language could reach a real spoof-vs-bonafide measurement in the time
available. A genuine-speech-only check for Marathi was already run *before*
this decision (see "Marathi (out of scope)" below) and is kept for
reference, not as tracked scope going forward.

## Status (honest, as of this write-up)

**What this pipeline can do without any human volunteers**, using existing
licensed public speech corpora instead of recruiting classmates:

- Real, consented genuine speech in **Hindi** — see "Datasets" below.
- Real synthetic (spoof) speech in **Hindi** via Coqui XTTS-v2
  (`generate_synthetic_corpus.py`) AND, as of 2026-09-08, a second,
  independent system, Resemble AI's Chatterbox
  (`generate_synthetic_corpus_chatterbox.py`) — see "Held-out
  generalisation check" below.
- A held-out-generator evaluation harness that runs the same
  `AasistAcousticDetector` the live service uses.

**Real bonafide-vs-spoof measurement for Hindi** (see `results/
summary.json`, run via `scripts/run_held_out_eval.py` against 200 genuine
Hindi utterances and 40 XTTS-v2-cloned Hindi utterances — scaled up
2026-09-08 from an original 20+10, see "Datasets" for the source corpus):

| | Bonafide accuracy | Spoof accuracy | EER |
|---|---|---|---|
| **Before fine-tuning** | 82% | 82.5% | **17.75%** |
| **After fine-tuning** (final layer only, 260 examples, class-balanced loss) | 100% | 92.5% | **3.0%** |

**Read this honestly — this result required finding and fixing two real
bugs, not just scaling up data:**

1. **Class imbalance, first attempt**: scaling genuine Hindi 10x (20→200)
   while only scaling synthetic 2x (20→40) left training 5.5:1 imbalanced.
   An unweighted loss plus selecting the "best" checkpoint by raw
   validation accuracy let the model collapse toward always predicting
   "bonafide" — measured directly: bonafide hit 100% but spoof fell to
   25% (worse than before fine-tuning), and EER got slightly WORSE
   (17.75%→19.75%), not better — a threshold shift disguised as an
   improvement by an accuracy metric the imbalance was gaming. Fixed with
   a class-weighted loss and selecting the best checkpoint by BALANCED
   (per-class-averaged) accuracy instead of raw accuracy.

2. **BatchNorm/Dropout statistics drift, second attempt (more
   fundamental)**: AASIST has 18 BatchNorm1d/2d layers plus Dropout in
   the "frozen" backbone. Setting `requires_grad=False` stops their
   WEIGHTS from updating but does nothing to stop BatchNorm's running
   statistics from drifting (they update via a momentum rule on every
   forward pass in `.train()` mode, independent of gradients) or Dropout
   from injecting fresh random noise, also only in `.train()` mode.
   Verified by hand: the exact same saved `out_layer` weights gave
   WILDLY different predictions live during training (drifted internal
   state) vs. reloaded fresh (base checkpoint's original state) —
   e.g. one held-out spoof example read spoof_probability~1.0 live but
   0.39 after reloading. Every number from both the un-fixed run and the
   imbalance-only-fixed run was measuring this never-saved, never-
   reproducible drifted state, not the weights actually persisted to
   disk. Fixed the simplest correct way: since `out_layer` is a bare
   `nn.Linear` with no train/eval-mode-dependent behaviour of its own,
   the whole model now stays in `.eval()` mode throughout training —
   `.eval()` only changes BatchNorm/Dropout's forward behaviour, it
   doesn't disable gradient flow to `out_layer`.

With both fixed, the result above is genuinely reproducible (verified: a
fresh reload through the exact production code path,
`AasistAcousticDetector.score()`, gives the same numbers `run_held_out_
eval.py` reports) — see `scripts/fine_tune_aasist.py`'s own REAL BUG
comments for the full account, kept rather than deleted so this doesn't
happen again.

## Held-out generalisation check (2026-09-08): does the recalibration actually generalise?

Every number above is measured against XTTS-v2 — the SAME synthesis
system the fine-tune trained on. That answers "did the mechanism improve
fit", not "does this generalise to a spoof system it's never seen" — the
blueprint's held-out-generator design has needed a second, independent
system for this the whole time. **Resemble AI's Chatterbox** (MIT,
`generate_synthetic_corpus_chatterbox.py`, run under its own dedicated
venv — see that script's docstring for why) is a genuinely different
model family and training data from XTTS-v2, and its output was NEVER
used in fine-tuning:

| | Bonafide accuracy | Spoof accuracy | EER |
|---|---|---|---|
| **Before fine-tuning** (XTTS-v2, trained-on) | 82% | 82.5% | **17.75%** |
| **After fine-tuning** (XTTS-v2, trained-on) | 100% | 92.5% | **3.0%** |
| **After fine-tuning, HELD-OUT** (Chatterbox, never seen) | 100% | 75% | **10.25%** |

**Read this honestly**: the held-out EER (10.25%) is worse than the
trained-on EER (3.0%) — expected, not a red flag; a model always does
better on the exact distribution it was calibrated against than on a
genuinely novel one. What matters is the comparison against the
BEFORE-fine-tuning baseline (17.75%): the recalibration cuts the held-out
EER nearly in half (17.75%→10.25%) against a synthesis system it never
trained on, which is real evidence this generalises to some degree,
not just memorised XTTS-v2's specific artifacts. It is also honestly not
as strong a result as the trained-on number, and 40 held-out spoof
examples (from 4 cloned reference speakers) is still a calibration-scale
sample, not a statistically solid one.

**What's still pending for the English + Hindi scope:**

- **An independently-measured English baseline EER** against a public
  benchmark split — AASIST currently runs on its published checkpoint,
  unvalidated by this team on English (see `docs/architecture.md`, "The
  acoustic detector").
- **More held-out spoof volume** — 40 examples (4 reference speakers ×
  10 sentences) from Chatterbox is a real, independent check, but still
  small. More reference speakers and sentences from either system would
  make the held-out EER a statistically solid number rather than a
  calibration-scale one.

## 2026-09-09 update: results on the expanded, properly-licensed corpus

Everything above (200 genuine / 40 spoof) is now superseded by a much
larger and more diverse corpus — see `TODO.md`'s Phase 1-4 for how it was
built: **1423 genuine Hindi** (existing IndicTTS-Hindi + Kathbath, CC BY
4.0 + Movie-MUSNOMIX, CC0/non-commercial + the 100 local speakers' own
training clips) and **340 spoof Hindi** (existing XTTS-v2 handful + new
XTTS-v2 clones of all 100 local speakers). Re-run via `scripts/
fine_tune_aasist.py` then `scripts/run_held_out_eval.py` (now itself
extended to cover the same expanded sources — see that script's own
`main()`):

| | Bonafide accuracy | Spoof accuracy | EER |
|---|---|---|---|
| **Before fine-tuning** | **36.5%** | 91.8% | **36.1%** |
| **After fine-tuning** (trained-on) | 85.8% | 90.9% | **12.0%** |
| **After fine-tuning, HELD-OUT synthesis system** (Chatterbox) | 85.8% | 82.5% | **15.0%** |
| **After fine-tuning, HELD-OUT on BOTH dimensions** (local speakers' own reserved test clips x Chatterbox) | 79.0% | 82.5% | **19.5%** |

**Read the "before" number honestly — it did not get worse; the test got
harder, and more honest.** The old 82% bonafide accuracy / 17.75% EER
baseline was measured against 200 clean IndicTTS-Hindi studio
recordings. This corpus adds Kathbath's real crowd-sourced audio and
Movie-MUSNOMIX's noisy movie dialogue — the SAME real-world gap the
"Informal spot-check" note below already flagged informally (75%
false-positive rate on an ad-hoc, unlicensed sample) is now measured
with properly-licensed, committed data instead: 36.5% bonafide accuracy,
i.e. the un-recalibrated English-trained AASIST checkpoint misclassifies
the majority of real, diverse Hindi speech as spoof. This is the honest
baseline the recalibration is actually solving, not a regression.

**The recalibration result is real and substantial at this larger
scale**: EER drops 36.1%→12.0% on trained-on data, and — the number that
actually matters — **36.1%→19.5% on data held out on BOTH the speaker/
utterance AND the synthesis-system dimensions at once** (never-seen
local-speaker test clips, never-seen Chatterbox spoof). That's the most
rigorous check this pipeline can currently run, and it still shows the
recalibration cutting the baseline gap by nearly half against data it
genuinely never saw in any form.

**Read this honestly too**: 100 held-out genuine + 40 held-out spoof
(the dual-held-out row) is still a calibration-scale sample, and 4 of
the 100 local speakers' test clips (`Nimish_1/2/4/5`) abstained as
near-silent rather than scoring — correctly handled (no fabricated
score), but a real gap in that speaker's usable held-out data. See
`scripts/run_held_out_eval.py`'s own comments for the exact
methodology, and `results/summary.json` for the full machine-readable
numbers.

### Informal spot-check (2026-09-08): the real gap is larger than the studio-recorded corpus above suggests

The before-fine-tune numbers above come from IndicTTS-Hindi — clean,
studio-quality read speech. A user-supplied, larger and more varied set
of real Hindi recordings (100 speakers, ordinary phone-/laptop-mic
quality, not studio-read) gave a very different, worse result when
spot-checked locally against the exact same shipped (un-recalibrated)
AASIST checkpoint: **75% false-positive rate** on 20 real genuine
speakers sampled from it (mean spoof-probability 0.73 — i.e. the model
is confidently, not just narrowly, wrong most of the time on this harder
set). Applying the fine-tuned output layer NOW ACTUALLY WIRED INTO
PRODUCTION (see above — the correctly-fixed one, not either of the two
earlier broken attempts) to the same 20 speakers: **20% false-positive
rate**, mean spoof-probability 0.25 — a real, substantial improvement on
a genuinely held-out set the fine-tuning never saw. (An earlier version
of this note reported 60% here, from a fine-tuned checkpoint produced
before the class-imbalance and BatchNorm/Dropout-drift bugs above were
found and fixed — that number is superseded, not additional evidence.)

**This finding is not part of the committed pipeline and never will be
using this specific data**: the source dataset
(`github.com/shivam-shukla/Speech-Dataset-in-Hindi-Language`) has **no
license declared** (confirmed via the GitHub API: `license: null`, no
LICENSE file, no license section in its README) — same situation as the
GramVaani corpus below, and the same rule applies: not usable in
`fetch_genuine_corpus.py`, fine-tuning, or any reported/shipped result. The
two numbers above were measured locally, once, against the user's own
already-downloaded copy, and are recorded here only as a directional
finding (studio TTS-adjacent read speech understates the real-world
Hindi gap) — not as a dataset this project has adopted.

**Actionable takeaway, updated**: scaling genuine Hindi data to real
volume (20→200, from the same properly-licensed IndicTTS-Hindi corpus)
plus fixing the two real fine-tuning bugs above IS what closed most of
this gap (75%→20%). What's left is now specifically **spoof-side**
volume and diversity — only 40 synthetic examples from 4 cloned
speakers exist; a second, held-out synthesis system (still pending
above) would both fix that and finally answer the generalisation
question honestly, rather than more scripts against this one unlicensed
corpus.

### Marathi (out of scope)

Genuine-only numbers exist as a byproduct of work done before the scope
decision above (no synthetic side — no viable ungated Marathi TTS system
found): 95% bonafide accuracy before fine-tuning, 100% after. This isn't
tracked as pending work; Marathi is not required scope.

## Prosodic detector calibration (2026-09-09): a real, concerning finding

`scripts/calibrate_prosodic_thresholds.py` ran Praat's real jitter/
shimmer/HNR extraction (the exact function `ParselmouthProsodyDetector`
calls in production) across every genuine and spoof file collected so
far — 1423 genuine train, 340 spoof train, 100 genuine held-out (local
speakers' own reserved test utterances), 40 spoof held-out (Chatterbox,
never used anywhere above). Full numbers: `results/
prosodic_calibration.json`.

**The currently-deployed thresholds score exactly 50% balanced
accuracy — chance level**, on both train and held-out data: 100%
genuine accuracy, 0% spoof accuracy. They never flag anything as spoof
against real data. This isn't a threshold-placement problem: genuine and
spoof jitter/shimmer/HNR distributions are nearly IDENTICAL in this
corpus (mean jitter 0.0203 genuine vs. 0.0226 spoof; shimmer 0.096 vs.
0.093; HNR 13.4dB vs. 13.7dB) — a percentile-based recalibration of the
same three independent floors/ceiling barely moves the needle (still
~50% balanced accuracy).

**A logistic regression combining all three features (standardized,
class-weighted) does meaningfully better: 82.5% balanced accuracy on
the held-out set** (70% genuine / 95% spoof) — a real, substantial
improvement over chance, using the same three Praat-derived numbers, just
combined and weighted rather than thresholded independently. Read this
honestly too: 140 held-out examples is still a calibration-scale sample,
and the learned coefficients (jitter +0.30, shimmer -0.32, HNR +0.037 on
standardized features) don't all point the direction the current
heuristic assumes — notably shimmer's sign is reversed from
`_SHIMMER_FLOOR`'s "low shimmer is suspicious" hypothesis on this
specific cloning system's artifacts.

**Not yet decided**: whether to replace `ParselmouthProsodyDetector`'s
hand-tuned formula with this trained classifier in production. That's an
architecture change (a serialized model behind the same `DetectorPort`),
not a config-constant edit — tracked in `TODO.md`'s Phase 5, pending an
explicit decision, not done unilaterally here.

## Datasets — verified before adopting (see `docs/risk-model.md`'s standing
discipline: real license/fetchability verification before adoption)

| Corpus | Language | License | Verified how |
|---|---|---|---|
| [SPRINGLab/IndicTTS-Hindi](https://huggingface.co/datasets/SPRINGLab/IndicTTS-Hindi) | Hindi | **CC BY 4.0** | Dataset card cites the original Indic TTS license; cross-checked against the [IndicTTS23 paper](https://arxiv.org/html/2410.14197v1), which states explicitly: "License: CC BY 4.0" and documents informed consent ("A consent form is obtained from the voice talents by confirming their acceptance to... release it under CC BY 4.0 license"). Confirmed ungated via the HF API. |
| [SPRINGLab/IndicTTS_Marathi](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Marathi) | Marathi | Same source project (Indic TTS Database, IIT Madras) | Confirmed ungated. **Caveat, not yet resolved:** this dataset's own README card contains a copy-pasted line saying "This HuggingFace dataset specifically contains the Hindi monolingual portion" — almost certainly a template artifact from reusing the Hindi card, but `fetch_genuine_corpus.py` verifies actual language content (see below) before trusting it, rather than taking the README at face value. |
| [ai4bharat/Kathbath](https://huggingface.co/datasets/ai4bharat/Kathbath) | Hindi | **CC BY 4.0** | Confirmed via the HF dataset card's own `cardData.license`/`tags` (`license:cc-by-4.0`) — but the HF-hosted copy is `gated: auto`, which this project's "no auth needed" discipline treats as a hard blocker regardless of license (see the "Rejected options" note below on Mann-ki-Baat/Spoken-Tutorial for the same rule). Fetched instead via AI4Bharat's own [`IndicSUPERB`](https://github.com/AI4Bharat/indicSUPERB) GitHub repo, which links the exact same audio, ungated, direct-download via E2E Networks object storage. `scripts/fetch_kathbath_hindi.py` streams the (12-language-bundled) clean/valid tar and keeps only the Hindi members — 500 utterances fetched (2026-09-09). |
| [mrinmoy-iitg/Movie-MUSNOMIX](https://github.com/mrinmoy-iitg/Movie-MUSNOMIX) | Hindi | **CC0 1.0 for the annotations only** — the repo's own README states the underlying movie audio (extracted from 4 YouTube-hosted Bollywood films, credited to their official channels) is restricted to **academic/non-commercial use only** ("According to the Copyright Act, 1957 of the Govt. of India, the content in this data corpus may only be used for academic research. No one is permitted to use this dataset commercially.") | Read the repo's own LICENSE (CC0) and README disclaimer directly (not just a dataset-card tag). This project is non-commercial, and the user explicitly confirmed (2026-09-09) that qualifies. Only the "speech" class (203 of 1161 total annotated segments; the rest are music/noise/mixed) was fetched via `scripts/fetch_movie_musnomix.py` — real Bollywood dialogue from 4 films (1966-2011), a genuinely different acoustic domain (dramatic delivery, background music bleed) from both IndicTTS-Hindi and Kathbath. **This data must never be used for anything commercial** — track that if this project's status ever changes. |

### Rejected options (and why)

- **AI4Bharat IndicF5** — gated on HuggingFace (`gated: "auto"`), same
  rejection reason already applied to pyannote.audio and Resemblyzer
  elsewhere in this project: it would require every user to manage an HF
  auth token, breaking the "no auth needed" build discipline. Also doesn't
  cover Marathi.
- **AI4Bharat Indic Parler TTS** — also gated (`gated: "auto"`), same
  rejection. Also doesn't cover Marathi (`en, as, bn, gu, hi, kn, ks, or, ml`
  — no `mr`).
- **Coqui XTTS-v2 for Marathi** — XTTS-v2 supports 17 languages including
  Hindi (added in v2.0.3) but not Marathi. Used for Hindi only.

### Under investigation — not yet adopted

- **[TheAIchemist13/gramvaani_preprocessed_hi_train](https://huggingface.co/datasets/TheAIchemist13/gramvaani_preprocessed_hi_train)**
  — real, natural (not studio-read) Hindi speech: 37,080 examples, ~15GB
  compressed, from GramVaani/Mobile Vaani's community-journalism audio.
  Ungated, but **its dataset card has no license field at all**
  ("[More Information needed]") — unlike the two corpora above, this has
  not been verified and is not part of the eval pipeline. 10 sample
  utterances were fetched to `data/unverified_license/gramvaani_hi/` for
  manual inspection only (gitignored, ~3MB) — real, natural field
  recordings (community-news style content, not read speech), which would
  be a genuinely useful complement to IndicTTS-Hindi's studio recordings
  if the license clears. **Do not use this data in the committed pipeline
  (fetch_genuine_corpus.py, fine-tuning, or anything reported as a result)
  until its licence is actually confirmed** — the original GramVaani ASR
  Challenge terms would be the place to check next.

## Sentence set

`prompts/sentences.md` — 10 fixed sentences (English source + Hindi +
Marathi), themed around the actual attack scenario this project defends
against (urgency, financial requests, claimed authority) rather than
neutral text, so prosody under "pressure" is represented.

## Consent

`CONSENT_FORM.md` — the one-page form the blueprint's §04 calls for,
ready for whenever a real volunteer (starting with Malvi) is recorded.
Nothing in this pipeline records a real person without it.

## Running it

```bash
cd eval/indian_language
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/fetch_genuine_corpus.py       # downloads a subset of the two datasets above, verifies language content
python scripts/generate_synthetic_corpus.py  # runs XTTS-v2 on the Hindi sentences against the genuine speakers' voices
                                              # (~2GB one-time model download, then ~10-15s/sentence on CPU)

# The next two steps import AasistAcousticDetector directly from
# services/live-call-api — run them with THAT service's own venv active,
# not this pipeline's venv (avoids a second multi-GB torch install):
source ../../services/live-call-api/.venv/bin/activate
python scripts/run_held_out_eval.py          # runs AASIST over genuine+synthetic, writes results/summary.json
python scripts/fine_tune_aasist.py           # fine-tunes AASIST's final layer, writes results/aasist_out_layer_finetuned.pth
python scripts/run_held_out_eval.py          # re-run: now reports before_finetune AND after_finetune
```

`data/`, `results/*.wav`, and `results/*.pth` are gitignored — only
`results/summary.json` (numbers, no audio, no model weights) is committed,
so this reproduces on any machine without re-downloading multi-gigabyte
corpora/models into git.
