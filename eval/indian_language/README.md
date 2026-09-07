# Indian-language dataset & held-out evaluation (blueprint §04)

This is the offline pipeline that produces the "Indian-language gap" numbers
referenced throughout `docs/blueprint.html` (§00, §04, §08) — it never
touches the live-call runtime in `services/live-call-api/`. See
`docs/architecture.md`'s second diagram ("Build-time — the Indian-language
dataset & evaluation pipeline") for how this fits the overall project.

## Status (honest, as of this write-up)

**What this pipeline can do without any human volunteers**, using existing
licensed public speech corpora instead of recruiting classmates:

- Real, consented genuine speech in **Hindi** and **Marathi** — see
  "Datasets" below.
- Real synthetic (spoof) speech in **Hindi** via Coqui XTTS-v2.
- A held-out-generator evaluation harness that runs the same
  `AasistAcousticDetector` the live service uses.

**First real measurement** (genuine speech only, no synthetic side yet —
see `results/summary.json`, run via `scripts/run_held_out_eval.py` against
20 real Hindi + 20 real Marathi utterances from the datasets below):

| Language | Genuine utterances scored | Correctly recognised as bonafide (not flagged as spoof) |
|---|---|---|
| Hindi | 20 | **85%** (mean spoof-probability 0.15) |
| Marathi | 20 | **95%** (mean spoof-probability 0.07) |

Read carefully: this is AASIST's *false-positive rate* on real Indian-
language speech it was never trained on — not yet the full bonafide-vs-
spoof accuracy the blueprint's §04/§08 wants, which needs the synthetic
(spoof) side too (see "Not started" below). It's still a real, useful
signal on its own: a 15% false-positive rate on genuine Hindi callers is
the kind of number that would actually matter to a bank deciding whether
to trust this system for Hindi-speaking customers, and it's honestly worse
than Marathi's 5% — small sample (n=20 each), not yet statistically
rigorous, but real and reproducible on demand, not claimed.

**What still needs a human, and hasn't happened yet:**

- **Malvi** — no public dataset or open TTS/cloning system exists for this
  dialect at all (it's a spoken Rajasthani/Hindi variant with no
  standardised script). This genuinely needs a native-speaker volunteer to
  record from scratch — see `CONSENT_FORM.md`, ready for exactly that.
- **Marathi synthetic (spoof) speech** — no ungated open TTS/cloning system
  covering Marathi turned up in a real search (see "Rejected options"
  below). Until one is found, the held-out-generator design only fully runs
  for Hindi; Marathi currently only has a genuine-speech "does AASIST over-
  flag real Marathi as spoof" check, not a full spoof/bonafide comparison.
- **A second, held-out synthesis system for Hindi** — the blueprint's design
  needs *two* systems, one held out entirely from anything the acoustic
  detector is fine-tuned on. Only one (XTTS-v2) is wired in so far.
- **Fine-tuning AASIST** on this data — scaffolding only; not yet run.

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
python scripts/run_held_out_eval.py          # runs AASIST over everything, writes results/summary.json
```

`data/` and `results/*.wav` are gitignored — only `results/summary.json`
(numbers, no audio) is committed, so this reproduces on any machine without
re-downloading multi-gigabyte corpora into git.
