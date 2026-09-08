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
- Real synthetic (spoof) speech in **Hindi** via Coqui XTTS-v2.
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

**What's still pending for the English + Hindi scope:**

- **An independently-measured English baseline EER** against a public
  benchmark split — AASIST currently runs on its published checkpoint,
  unvalidated by this team on English (see `docs/architecture.md`, "The
  acoustic detector").
- **A second, held-out synthesis system for Hindi** — still missing.
  The blueprint's design needs *two* systems, one held out entirely from
  anything the acoustic detector is fine-tuned on. Only one (XTTS-v2) is
  wired in so far, so "after fine-tuning" is still only measured on data
  the model already saw (in whole or in part), not a true generalisation
  test — fixing the two bugs above made the EXISTING measurement
  trustworthy, it didn't add the second system this gap has always
  needed.
- **Only 40 spoof examples** — validation's spoof split is just 8
  examples (each worth 12 percentage points on val_spoof_acc); the
  headline numbers above are real and reproducible, but still a
  calibration-scale result, not a statistically solid one. More spoof
  volume (more reference speakers, more sentences, or the second
  synthesis system above) would fix this properly.

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

## Datasets — verified before adopting (see `docs/risk-model.md`'s standing
discipline: real license/fetchability verification before adoption)

| Corpus | Language | License | Verified how |
|---|---|---|---|
| [SPRINGLab/IndicTTS-Hindi](https://huggingface.co/datasets/SPRINGLab/IndicTTS-Hindi) | Hindi | **CC BY 4.0** | Dataset card cites the original Indic TTS license; cross-checked against the [IndicTTS23 paper](https://arxiv.org/html/2410.14197v1), which states explicitly: "License: CC BY 4.0" and documents informed consent ("A consent form is obtained from the voice talents by confirming their acceptance to... release it under CC BY 4.0 license"). Confirmed ungated via the HF API. |
| [SPRINGLab/IndicTTS_Marathi](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Marathi) | Marathi | Same source project (Indic TTS Database, IIT Madras) | Confirmed ungated. **Caveat, not yet resolved:** this dataset's own README card contains a copy-pasted line saying "This HuggingFace dataset specifically contains the Hindi monolingual portion" — almost certainly a template artifact from reusing the Hindi card, but `fetch_genuine_corpus.py` verifies actual language content (see below) before trusting it, rather than taking the README at face value. |

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
