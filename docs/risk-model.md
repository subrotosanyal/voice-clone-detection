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

## What's still a placeholder (read before trusting a score)

The prosodic detector (`prosody_pitch_variance.py`) still computes a real
signal-processing feature (autocorrelation-based pitch variance) that is
**not** a validated indicator of synthetic speech on its own — its
docstring says so, and names the honest next step (a Parselmouth/Praat
front-end or a learned model). The lightweight acoustic fallback
(`acoustic_spectral_flatness.py`, still available via a one-line config
swap) has the same caveat. Treat any score that includes either of these
as partly proof-of-pipeline, not a fully validated fraud signal — the
acoustic-via-AASIST component is the one part of today's score backed by
a real, published, trained model.

## The pluggable third signal, pros/cons

| Mode | Pros | Cons |
|---|---|---|
| `contextual` | No ML/enrollment infra; zero privacy burden; legible to a non-technical reader; works from the first call | Weak alone — an attacker who knows the target's number and calls in business hours evades it entirely |
| `consistency` | Checks identity directly; harder to defeat even for a sophisticated attacker | Needs an enrollment flow, consent record, embedding store, TTL — real extra build; cold-start for anyone not enrolled |
| `auto` (default) | Consistency when available, contextual fallback otherwise — no silent zero | Slightly more moving parts to reason about than a fixed mode |

Switch modes in `config/risk_formula.yaml` → `detectors[third_signal].params.mode`
— no code change.
