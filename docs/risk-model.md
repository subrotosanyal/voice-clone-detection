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

## What's a placeholder right now (read before trusting a score)

The acoustic and prosodic detectors compute real signal-processing
features (spectral flatness; autocorrelation-based pitch variance) — they
are **not** validated indicators of synthetic speech on their own. Every
detector module says this explicitly in its docstring, and the honest
next step for each is named there (AASIST/RawGAT-ST for acoustic; a
Parselmouth/Praat front-end or a learned model for prosody). Treat every
score from this repo today as a proof that the *pipeline* works end to
end — not as a validated fraud signal.

## The pluggable third signal, pros/cons

| Mode | Pros | Cons |
|---|---|---|
| `contextual` | No ML/enrollment infra; zero privacy burden; legible to a non-technical reader; works from the first call | Weak alone — an attacker who knows the target's number and calls in business hours evades it entirely |
| `consistency` | Checks identity directly; harder to defeat even for a sophisticated attacker | Needs an enrollment flow, consent record, embedding store, TTL — real extra build; cold-start for anyone not enrolled |
| `auto` (default) | Consistency when available, contextual fallback otherwise — no silent zero | Slightly more moving parts to reason about than a fixed mode |

Switch modes in `config/risk_formula.yaml` → `detectors[third_signal].params.mode`
— no code change.
