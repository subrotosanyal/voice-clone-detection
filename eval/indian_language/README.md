# Indian-language dataset & held-out evaluation (blueprint §04)

This is the offline pipeline that produces the "Indian-language gap" numbers
referenced throughout `docs/blueprint.html` (§00, §04, §08) — it never
touches the live-call runtime in `services/live-call-api/`. See
`docs/architecture.md`'s second diagram ("Build-time — the Indian-language
dataset & evaluation pipeline") for how this fits the overall project.

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

**First real bonafide-vs-spoof measurement for Hindi** (see
`results/summary.json`, run via `scripts/run_held_out_eval.py` against 20
genuine Hindi utterances and 10 XTTS-v2-cloned Hindi utterances):

| | Bonafide accuracy | Spoof accuracy | EER |
|---|---|---|---|
| **Before fine-tuning** | 85% | 100% | **10%** |
| **After fine-tuning** (final layer only, 50 examples) | 100% | 80% | **10%** |

**Read this honestly, not optimistically:** fine-tuning did NOT reduce the
EER — it moved AASIST's decision threshold, catching every genuine Hindi
speaker correctly (85%→100%) at the direct cost of missing more of the
cloned speech (100%→80% spoof accuracy). The Equal Error Rate — the
number that actually measures how separable bonafide and spoof are for
this model — stayed exactly at 10% before and after. That's the honest
finding: **with only 50 total examples, fine-tuning the final layer
re-balances where errors land, it doesn't make the model more capable of
telling real from fake Hindi speech.** Real improvement needs real
volume — more speakers, more sentences, and critically, a second held-out
synthesis system so "after fine-tuning" can be measured on data the model
never touched, instead of the same data it trained on (see
`fine_tune_aasist.py`'s docstring on why only the last layer was
fine-tuned at all, given this data volume).

**What's still pending for the English + Hindi scope:**

- **An independently-measured English baseline EER** against a public
  benchmark split — AASIST currently runs on its published checkpoint,
  unvalidated by this team on English (see `docs/architecture.md`, "The
  acoustic detector").
- **A second, held-out synthesis system for Hindi** — the blueprint's design
  needs *two* systems, one held out entirely from anything the acoustic
  detector is fine-tuned on. Only one (XTTS-v2) is wired in so far, so
  "after fine-tuning" is only measured on data the model already saw, not a
  real generalisation test.
- **More Hindi data** — the fine-tune ran on only 50 total examples; more
  speakers and sentences from the same verified corpus (see "Datasets"
  below) would make the EER number statistically meaningful rather than an
  artifact of a 30-example test set.

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
