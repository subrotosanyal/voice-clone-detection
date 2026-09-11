// Satya-Vani — Live Call Path — browser dashboard.
// No build step, no framework: this talks to the same two endpoints a
// curl command would (POST /v1/score/file and WS /v1/stream/{session_id}),
// documented in docs/architecture.md. If you're reading this to understand
// the wire format, those two routes (app/api/http_router.py,
// app/api/ws_router.py) are the source of truth.

// Set by index.html from settings.base_path (see app/main.py's "/" route
// and app/config.py's own note) — "" unless this app is path-routed under
// a prefix on a shared domain. Every fetch()/WebSocket URL below is built
// from this instead of a bare leading "/", so it still resolves against
// the right origin+prefix when Traefik strips the prefix back off before
// forwarding to this container.
const BASE_PATH = (window.__BASE_PATH__ || "").replace(/\/+$/, "");

const BAND_COLOR = { low: "var(--risk-low)", elevated: "var(--risk-med)", high: "var(--risk-high)" };
const COMPONENT_LABEL = {
  acoustic: "Acoustic (AASIST)",
  prosodic: "Prosodic",
  third_signal: "Third signal",
  intent: "Intent (zero-shot)",
  perth_watermark: "Watermark check",
  phase_incoherence: "Phase coherence",
  semantic_risk: "Semantic risk (local LLM)",
};
// Friendly names for FusedScoreOut.transcript_source (a transcriber's own
// `detector_name` — see routing_transcriber.py for how a single call can
// end up at either one). Falls back to the raw value for anything not
// listed here, so a future transcriber never renders as blank.
const TRANSCRIBER_LABEL = {
  whisper_transcriber: "Whisper",
  vexyl_stt_transcriber: "VEXYL-STT (Indic)",
  whisperlive_transcriber: "WhisperLive",
  routing_transcriber: "Routing",
};
const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 62;

// ---------- help tooltips — plain-language explanations for jargon in the UI ----------
const COMPONENT_HELP = {
  acoustic: "Detects whether the VOICE ITSELF was AI-generated or cloned, as opposed to a real person speaking — using AASIST, a neural network trained specifically to spot the audio artifacts text-to-speech and voice-cloning tools leave behind that a real human voice doesn't have. Higher % = more likely synthetic/cloned. The base model is only independently validated on English speech; a separately fine-tuned final layer improves accuracy on Hindi specifically (see 'view raw JSON' for whether it was active).",
  prosodic: "Detects whether the voice's pitch and loudness variation (jitter, shimmer) and harmonics-to-noise ratio look natural — using a logistic regression trained on 1423 genuine + 340 spoof Hindi examples, not a hand-picked threshold. Higher % = more spoof-like on these three measures. Reaches 82.5% balanced accuracy on genuinely held-out data (2026-09-09) — a real, substantial improvement over an earlier hand-tuned version of this detector, which turned out to score at chance level against real audio once actually measured.",
  third_signal: "A third, swappable check — NOT about the audio itself. Either (a) red flags about the CALL: an unknown number, an odd hour, urgent/pressuring language, an authority claim (bank/police/government) — especially combined with a financial request, the classic fraud script — or (b) a direct voice match check against a caller's enrolled voiceprint (Voiceprints tab), when one exists. Whichever ran is named in 'view raw JSON'.",
  intent: "Scores the TRANSCRIPT (not the audio) against fraud-relevant candidate labels — 'requesting a money transfer', 'requesting an OTP/PIN', 'impersonating a bank or government official', 'creating urgency', or 'ordinary conversation' — using a zero-shot language model, not exact keyword matching like Third signal. KNOWN LIMITATION: testing found it sometimes misjudges completely ordinary conversation as suspicious — the breakdown below shows every candidate's own score, not just the winner, specifically so you can sanity-check it rather than trust one number blindly. Weighted low in the overall score for that reason. Abstains with no transcript.",
  perth_watermark: "Checks for a specific neural fingerprint (the 'Perth' watermark) that Chatterbox and other Resemble AI-based voice-cloning tools embed in every clip they generate — a narrow but high-confidence check, not a general spoof detector. Higher % = this specific fingerprint was found. NARROW SCOPE: it only catches tools that use this watermark — a low score here does NOT mean the audio is genuine, it just means this one fingerprint wasn't found; see Acoustic for general-purpose spoof detection. Known false-positive on non-speech audio like a pure tone.",
  phase_incoherence: "Checks how consistent the audio's frequency PHASE is from one instant to the next — AI voice generators tend to reconstruct a cleaner, more mathematically 'tidy' phase than a real human voice, which is a noisier physical process. Higher % = unusually phase-coherent for genuine speech. KNOWN LIMITATION: this is a moderate signal against XTTS-v2-style cloning but a much stronger one against Chatterbox-style cloning — testing found it genuinely weaker against some cloning tools than others, so treat a low score here as 'no strong phase evidence', not 'definitely genuine'. Weighted lowest of the acoustic checks for that reason. A real, deterministic measurement (like Prosodic), not a trained classifier like Acoustic.",
  semantic_risk: "Reads the TRANSCRIPT through a small language model running locally (no cloud call) — asked to directly judge the conversation for manufactured urgency, financial requests, claimed authority, or a request not to hang up/tell anyone, rather than scoring fixed labels like Intent does. Runs ALONGSIDE Intent, not in place of it: testing found it correctly handles an ordinary sentence Intent misjudges as suspicious, and correctly flags real scam scripts including the 'don't hang up' tactic neither Intent nor Third signal's keyword rules currently catch. Still only verified on a handful of examples, not a large held-out evaluation. Abstains with no transcript.",
};
const BAND_HELP = {
  low: "LOW risk (score 0–34): nothing here looks suspicious across the signals that ran. Recommended action: no special handling needed.",
  elevated: "ELEVATED risk (score 35–69): at least one signal flagged something worth a closer look. Recommended action: verify the caller further before acting on a sensitive request.",
  high: "HIGH risk (score 70–100): multiple signals agree something is off, or one signal is very confident. Recommended action: block or escalate the sensitive request for manual review.",
};

// ---------- "How this score is calculated" — full pictorial breakdown ----------
// Kind + step-by-step mechanics for each detector, shown in the How It Works
// modal alongside its LIVE weight (pulled from /v1/config, never hardcoded
// here, so this page can't drift from the real running formula).
const HIW_DETECTOR_INFO = {
  acoustic: {
    kind: "🧠 Trained neural network — AASIST (ASVspoof2019 LA)",
    steps: [
      "The raw waveform is fed directly into AASIST, a graph-attention network trained specifically to spot text-to-speech/voice-cloning artifacts.",
      "Output is a single spoof-probability between 0 and 1 — no hand-built features in between.",
      "A small, separately fine-tuned final layer improves accuracy on Hindi speech specifically.",
    ],
    caveat: "Only independently validated on English so far — Hindi relies on the fine-tuned layer above.",
  },
  prosodic: {
    kind: "🧠 Trained classifier — logistic regression over Praat/Parselmouth features",
    steps: [
      "Praat extracts jitter (pitch-period variability), shimmer (amplitude variability), and HNR (harmonics-to-noise ratio) — validated clinical voice-quality measures.",
      "The three numbers are standardized and combined by a logistic regression trained on 1423 genuine + 340 spoof Hindi examples (2026-09-09), not three independent hand-picked thresholds.",
      "Output is a single spoof probability between 0 and 1.",
    ],
    caveat: "82.5% balanced accuracy on genuinely held-out data (never used in training) — a real, substantial improvement over an earlier hand-tuned version, which scored at exactly chance level (50%) once actually measured against real audio. Still a 3-feature classifier on a calibration-scale (140-example) held-out set, not a large validated benchmark.",
  },
  third_signal: {
    kind: "🔀 Rule-based OR 🧠 trained model, depending on mode",
    steps: [
      "Contextual mode: checks call metadata (unknown number, odd hour) and transcript keywords (urgency, authority claims, financial requests) against a transparent rule list.",
      "Consistency mode: compares the live voice's ECAPA-TDNN embedding against an enrolled voiceprint via cosine similarity.",
      "'auto' mode (the default) uses consistency once a voiceprint is enrolled, contextual otherwise.",
    ],
    caveat: null,
  },
  intent: {
    kind: "🧠 Trained language model — zero-shot NLI",
    steps: [
      "The call transcript is scored against fraud-relevant candidate labels ('requesting a money transfer', 'impersonating a bank', etc.) using a zero-shot classifier — not keyword matching.",
      "The highest-scoring fraud-relevant label (excluding 'ordinary conversation') sets the risk score.",
      "Abstains entirely when no transcript is available.",
    ],
    caveat: "Testing found this model can misjudge ordinary conversation as fraud-relevant — weighted low for that reason, and every candidate label's own score is shown, not just the winner.",
  },
  perth_watermark: {
    kind: "📐 Deterministic fingerprint check",
    steps: [
      "Checks the audio for Resemble AI's Perth neural watermark — a specific fingerprint Chatterbox and other Perth-integrated cloning tools embed in every clip they generate.",
      "Returns how strongly that exact fingerprint is present, not a general 'sounds synthetic' judgment.",
    ],
    caveat: "A low score only means this ONE fingerprint wasn't found — never that the audio is genuine. Known false-positive on non-speech audio like a pure tone.",
  },
  phase_incoherence: {
    kind: "📐 Deterministic formula — STFT phase analysis",
    steps: [
      "Measures how consistent the audio's frequency phase is from one instant to the next (circular variance of frame-to-frame phase, weighted by which frequencies actually carry energy).",
      "Neural vocoders tend to reconstruct cleaner, more mathematically 'tidy' phase than a real human voice.",
      "Unusually phase-coherent audio raises the score.",
    ],
    caveat: "Meaningfully weaker against some cloning tools (XTTS-v2) than others (Chatterbox) — weighted lowest of the acoustic-family signals for that reason.",
  },
  semantic_risk: {
    kind: "🧠 Local generative language model — Phi-3-mini (runs on-device, no cloud call)",
    steps: [
      "The call transcript is read by a small, local LLM asked to judge it directly against explicit criteria, rather than scoring fixed candidate labels.",
      "Returns a manufactured-urgency level, plus whether it found a financial request, a claimed authority, or a request not to hang up / tell anyone.",
      "The risk score is the strongest of those signals — any one on its own can raise it, not just a combination.",
    ],
    caveat: "Runs alongside the intent detector above, not in place of it — verified by hand on a handful of examples (including the exact ordinary sentence the intent detector misjudges), not yet a large held-out evaluation.",
  },
};

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function renderTranscriptNote(detail) {
  if (!detail || !detail.transcript) return "";
  const keywords = detail.urgency_keywords_from_transcript || [];
  const authorityKeywords = detail.authority_keywords_from_transcript || [];
  const financial = detail.is_financial_request_from_transcript;
  // Live-mic path only — a naive whole-session concatenation (may repeat
  // a phrase at cycle boundaries, see live_transcription.py's own honesty
  // note), shown collapsed so the current rolling transcript above stays
  // the primary thing a viewer reads.
  const fullTranscript = detail.full_session_transcript;
  return `
    <div class="transcript-note">
      <div class="transcript-label">Transcript (auto-detected, feeds this signal)</div>
      <div class="transcript-text">"${escapeHtml(detail.transcript)}"</div>
      ${keywords.length ? `<div class="transcript-tags">urgency: ${keywords.map((k) => `<span class="tag">${escapeHtml(k)}</span>`).join(" ")}</div>` : ""}
      ${authorityKeywords.length ? `<div class="transcript-tags">authority claim: ${authorityKeywords.map((k) => `<span class="tag">${escapeHtml(k)}</span>`).join(" ")}</div>` : ""}
      ${financial ? `<div class="transcript-tags"><span class="tag owner">financial request detected</span></div>` : ""}
      ${fullTranscript && fullTranscript !== detail.transcript ? `
        <details class="full-transcript-details">
          <summary>Full call transcript so far (may repeat a phrase at boundaries)</summary>
          <div class="transcript-text">"${escapeHtml(fullTranscript)}"</div>
        </details>
      ` : ""}
    </div>
  `;
}

function renderExplanationNote(detail) {
  if (!detail || !detail.explanation) return "";
  return `<div class="explanation-note">${escapeHtml(detail.explanation)}</div>`;
}

// Shows EVERY candidate label's own score, not just the winner — the
// intent detector's whole point is that a false positive should be
// visible and inspectable here, not hidden inside one opaque number. See
// app/adapters/intent/zero_shot_intent_classifier.py's HONESTY NOTE.
function renderIntentBreakdown(detail) {
  if (!detail || !detail.label_scores) return "";
  const entries = Object.entries(detail.label_scores).sort((a, b) => b[1] - a[1]);
  const rows = entries
    .map(([label, score]) => {
      const pct = Math.round(score * 100);
      const isOrdinary = label === "ordinary conversation";
      return `
        <div class="intent-row${isOrdinary ? " intent-row-ordinary" : ""}">
          <span class="intent-label">${escapeHtml(label)}</span>
          <div class="intent-bar-track"><div class="intent-bar-fill${isOrdinary ? " ordinary" : ""}" style="width:${pct}%"></div></div>
          <span class="intent-pct mono">${pct}%</span>
        </div>
      `;
    })
    .join("");
  return `
    <div class="intent-breakdown">
      <div class="intent-breakdown-label">Every candidate the model considered (not just the top pick)</div>
      ${rows}
    </div>
  `;
}

function helpIcon(text) {
  const span = document.createElement("span");
  span.className = "help-icon";
  span.tabIndex = 0;
  span.textContent = "?";
  span.dataset.help = text;
  return span;
}

let traceScores = []; // rolling [0-100] OVERALL fused scores for the current session/analysis, drives the chart
let traceComponents = {}; // { [detectorName]: (number|null)[] } — each detector's own smoothed_score*100 per window, same index alignment as traceScores; null where that detector had no value yet (never scored, no history to carry forward)
let traceTimestamps = []; // window_start_ms per point, same index alignment as traceScores — drives the x-axis
let traceSignal = []; // (number|null)[] — raw audio-energy intensity per window (see extractSignalIntensityPct below), same index alignment as traceScores. A diagnostic aid, not a risk signal: lets a viewer tell "there was no signal here" apart from "there was signal but nothing was flagged", for both the live-mic and file-upload paths (this same rendering function serves both).
let chartFactorEnabled = {}; // { [detectorName]: boolean } — chart-legend toggle state, defaults to true the first time a name is seen; persists across re-analyses in this page session
let lastConfig = null; // most recent /v1/config response, cached so other UI text (e.g. the analysing message) can name the actual configured detectors instead of a hardcoded, driftable list

// Fixed categorical palette for the per-factor chart lines — distinct
// hues, chosen to read on both light and dark backgrounds (see
// index.html's :root / dark-media color tokens for the rest of the
// page's palette; these are deliberately NOT theme-swapped, since a
// stable identity per detector matters more here than exact contrast
// optimization, and all are mid-brightness/saturated enough for either
// ground). COMPONENT_CHART_FALLBACK_COLORS cycles for any detector name
// not listed (e.g. a custom one added via config swap).
const COMPONENT_CHART_COLOR = {
  acoustic: "#e0763a",
  prosodic: "#4f8fe0",
  third_signal: "#a367d9",
  intent: "#3aa89a",
  perth_watermark: "#d9457a",
  phase_incoherence: "#9ab83a",
  semantic_risk: "#d9b83a",
};
const COMPONENT_CHART_FALLBACK_COLORS = ["#888888", "#c76b3d", "#3d7bc7", "#7dbf5f"];
function componentChartColor(name) {
  if (COMPONENT_CHART_COLOR[name]) return COMPONENT_CHART_COLOR[name];
  const idx = Object.keys(COMPONENT_CHART_COLOR).length % COMPONENT_CHART_FALLBACK_COLORS.length;
  return COMPONENT_CHART_FALLBACK_COLORS[idx];
}
const SIGNAL_CHART_COLOR = "#8a97a8"; // deliberately neutral/slate — reads as "diagnostic", not a risk-factor color

// Reuses whichever component's own `detail.rms` is already present (every
// one of these computes the SAME window's RMS as part of its own scoring
// — see e.g. acoustic_aasist.py's floor_rms abstain check) rather than
// asking the backend for a new field: no API change needed for this.
// Preferred order just picks the first one this fusion config happens to
// have active; any of them carries the same rms value for a given window.
const SIGNAL_RMS_SOURCE_ORDER = ["acoustic", "prosodic", "perth_watermark", "phase_incoherence"];
function extractSignalIntensityPct(components) {
  for (const name of SIGNAL_RMS_SOURCE_ORDER) {
    const c = components.find((c) => c.name === name);
    const rms = c && c.detail && typeof c.detail.rms === "number" ? c.detail.rms : null;
    if (rms !== null) return rmsToIntensityPct(rms);
  }
  return null;
}

// dB-relative-to-full-scale mapping, not linear: real observed RMS spans
// roughly two orders of magnitude between "near-silent" (~0.002-0.01) and
// "clearly present speech" (~0.02-0.1+) — a linear 0-1 scale would
// compress nearly everything into the bottom few percent, defeating the
// whole point of a visual "is there signal here" indicator. Clamped
// range chosen to match those observed values: -60dBFS -> 0%,
// -10dBFS -> 100%.
function rmsToIntensityPct(rms) {
  const db = 20 * Math.log10(Math.max(rms, 1e-6));
  const pct = ((db + 60) / 50) * 100;
  return Math.max(0, Math.min(100, pct));
}

// ---------- "How this score is calculated" modal ----------
// Renders live from whatever /v1/config actually returns — if
// risk_formula.yaml changes (a weight, an added/removed detector, a
// smoothing alpha, a band threshold), this page updates automatically
// with it, rather than being a second, driftable copy of the numbers.
function renderHowItWorks(cfg) {
  const detectors = cfg.detectors || [];
  const weightSum = detectors.reduce((sum, d) => sum + d.weight, 0) || 1;

  const winMeta = document.getElementById("hiwWindowMeta");
  if (winMeta && cfg.windowing) {
    winMeta.textContent = `${cfg.windowing.window_ms / 1000}s / ${cfg.windowing.hop_ms / 1000}s hop`;
  }
  const countMeta = document.getElementById("hiwDetectorCount");
  if (countMeta) countMeta.textContent = `${detectors.length} pluggable signals`;

  const alpha = cfg.fusion && cfg.fusion.smoothing_alpha;
  const alphaMeta = document.getElementById("hiwAlphaMeta");
  if (alphaMeta && alpha != null) alphaMeta.textContent = `EMA, α=${alpha}`;

  const cardsEl = document.getElementById("hiwDetectors");
  if (cardsEl) {
    cardsEl.innerHTML = detectors
      .map((d) => {
        const info = HIW_DETECTOR_INFO[d.name] || { kind: "", steps: [], caveat: null };
        const pct = Math.round((d.weight / weightSum) * 100);
        const label = COMPONENT_LABEL[d.name] || d.name;
        return `
          <div class="hiw-card">
            <div class="hiw-card-top">
              <span class="hiw-card-name">${escapeHtml(label)}</span>
              <span class="hiw-card-weight">${pct}% weight</span>
            </div>
            <div class="hiw-kind">${escapeHtml(info.kind)}</div>
            <ol class="hiw-steps">${info.steps.map((s) => `<li>${escapeHtml(s)}</li>`).join("")}</ol>
            ${info.caveat ? `<div class="hiw-caveat">⚠ ${escapeHtml(info.caveat)}</div>` : ""}
          </div>
        `;
      })
      .join("");
  }

  const eqEl = document.getElementById("hiwFormulaEq");
  if (eqEl) {
    const terms = detectors
      .map((d) => `${((d.weight / weightSum) * 100).toFixed(0)}%×${COMPONENT_LABEL[d.name] || d.name}`)
      .join(" + ");
    eqEl.textContent = `fused = ${terms}  (renormalized to 100% over signals with a usable value — including one carried forward from an earlier window if this one abstained)`;
  }

  const smoothEl = document.getElementById("hiwSmoothingEq");
  if (smoothEl && alpha != null) {
    // Updated 2026-09-10: smoothing moved to PER-detector (was a single
    // EMA on the combined total) — see WeightedSumFusion's own docstring.
    // Each detector keeps its own smoothed value across windows; the
    // fused score above is the weighted combination of those, not a
    // second smoothing pass on top.
    smoothEl.textContent = `each detector: smoothed = ${alpha} × its own raw_this_window + ${(1 - alpha).toFixed(2)} × its own smoothed_previous_window (unchanged if it abstains this window) — fused above combines these smoothed values directly`;
  }

  const bandsEl = document.getElementById("hiwBands");
  if (bandsEl && cfg.bands) {
    const { low_max: lowMax, elevated_max: elevatedMax } = cfg.bands;
    const helpText = (s) => s.substring(s.indexOf(": ") + 2);
    bandsEl.innerHTML = `
      <div class="hiw-band-track">
        <div class="hiw-band-seg" style="width:${lowMax}%; background:var(--risk-low);">0–${lowMax}</div>
        <div class="hiw-band-seg" style="width:${elevatedMax - lowMax}%; background:var(--risk-med);">${lowMax + 1}–${elevatedMax}</div>
        <div class="hiw-band-seg" style="width:${100 - elevatedMax}%; background:var(--risk-high);">${elevatedMax + 1}–100</div>
      </div>
      <div class="hiw-band-legend">
        <div class="hiw-band-legend-row"><span class="dot" style="background:var(--risk-low);"></span><span class="range mono">0–${lowMax}</span><span>${escapeHtml(helpText(BAND_HELP.low))}</span></div>
        <div class="hiw-band-legend-row"><span class="dot" style="background:var(--risk-med);"></span><span class="range mono">${lowMax + 1}–${elevatedMax}</span><span>${escapeHtml(helpText(BAND_HELP.elevated))}</span></div>
        <div class="hiw-band-legend-row"><span class="dot" style="background:var(--risk-high);"></span><span class="range mono">${elevatedMax + 1}–100</span><span>${escapeHtml(helpText(BAND_HELP.high))}</span></div>
      </div>
    `;
  }
}

const hiwOverlay = document.getElementById("hiwOverlay");
function openHowItWorks() { hiwOverlay.classList.add("show"); }
function closeHowItWorks() { hiwOverlay.classList.remove("show"); }
document.getElementById("howItWorksBtn").addEventListener("click", openHowItWorks);
document.getElementById("howItWorksFootLink").addEventListener("click", (e) => { e.preventDefault(); openHowItWorks(); });
document.getElementById("hiwClose").addEventListener("click", closeHowItWorks);
hiwOverlay.addEventListener("click", (e) => { if (e.target === hiwOverlay) closeHowItWorks(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeHowItWorks(); });

// ---------- health chip ----------
async function checkHealth() {
  const chip = document.getElementById("healthChip");
  const text = document.getElementById("healthText");
  try {
    const res = await fetch(`${BASE_PATH}/healthz`);
    if (!res.ok) throw new Error("not ok");
    chip.classList.add("ok");
    text.textContent = "service healthy";
    const cfg = await (await fetch(`${BASE_PATH}/v1/config`)).json();
    lastConfig = cfg;
    document.getElementById("formulaVersion").textContent = cfg.formula_version;
    renderHowItWorks(cfg);
  } catch {
    chip.classList.remove("ok");
    text.textContent = "service unreachable";
  }
}
checkHealth();

// ---------- tabs ----------
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
    if (isRecording) stopMic(); // switching tabs mid-recording ends it cleanly
    if (btn.dataset.tab === "history") loadHistoryList();
    if (btn.dataset.tab === "enroll") loadEnrollmentList();
  });
});

// ---------- context fields (shared builder for both tabs) ----------
function buildContextFields(container) {
  container.innerHTML = `
    <div class="field checkbox"><input type="checkbox" class="f-known"><label>Caller's number is known / on file</label></div>
    <div class="field checkbox"><input type="checkbox" class="f-financial"><label>A financial action is being requested</label></div>
    <div class="field checkbox"><input type="checkbox" class="f-authority"><label>Caller claims to be a bank/police/government official</label></div>
    <div class="field">
      <label>Hour of day</label>
      <select class="f-hour"></select>
    </div>
    <div class="field">
      <label>Urgency language heard (comma-separated)</label>
      <input type="text" class="f-urgency" placeholder="e.g. immediately, don't tell anyone">
    </div>
    <div class="field">
      <label>Claimed caller identity</label>
      <input type="text" class="f-identity" placeholder="e.g. alice — matches a Voiceprints tab enrollment">
    </div>
  `;
  const hourSelect = container.querySelector(".f-hour");
  for (let h = 0; h < 24; h++) {
    const opt = document.createElement("option");
    opt.value = h;
    opt.textContent = String(h).padStart(2, "0") + ":00";
    hourSelect.appendChild(opt);
  }
  hourSelect.value = new Date().getHours();
}
document.querySelectorAll(".field-grid").forEach(buildContextFields);

function readContext(scopeEl) {
  const known = scopeEl.querySelector(".f-known").checked;
  const financial = scopeEl.querySelector(".f-financial").checked;
  const authority = scopeEl.querySelector(".f-authority").checked;
  const hour = parseInt(scopeEl.querySelector(".f-hour").value, 10);
  const urgencyRaw = scopeEl.querySelector(".f-urgency").value.trim();
  const urgency_keywords = urgencyRaw ? urgencyRaw.split(",").map((s) => s.trim()).filter(Boolean) : [];
  const identity = scopeEl.querySelector(".f-identity").value.trim();
  const context = {
    known_number: known,
    is_financial_request: financial,
    authority_claim: authority,
    hour_of_day: hour,
    urgency_keywords,
  };
  if (identity) context.claimed_identity = identity;
  return context;
}

// ---------- results rendering (shared by both modes) ----------
function resetResults() {
  traceScores = [];
  traceComponents = {};
  traceTimestamps = [];
  traceSignal = [];
  document.getElementById("resultsEmpty").style.display = "flex";
  document.getElementById("results").classList.remove("show");
  document.getElementById("speakersSection").style.display = "none";
  document.getElementById("speakerCards").innerHTML = "";
  hideAnalysing();
}

// ---------- "analysing" progress state (file-upload only — the mic tab
// already gives live per-window feedback as WS messages arrive) ----------
let analysingIntervalId = null;
let analysingStartMs = 0;

function showAnalysing(detailText) {
  document.getElementById("resultsEmpty").style.display = "none";
  document.getElementById("results").classList.remove("show");
  document.getElementById("analysingDetail").textContent = detailText;
  document.getElementById("analysingElapsed").textContent = "0.0s elapsed";
  document.getElementById("analysingState").classList.add("show");
  analysingStartMs = performance.now();
  if (analysingIntervalId) clearInterval(analysingIntervalId);
  analysingIntervalId = setInterval(() => {
    const elapsedS = (performance.now() - analysingStartMs) / 1000;
    document.getElementById("analysingElapsed").textContent = elapsedS.toFixed(1) + "s elapsed";
  }, 100);
}

function hideAnalysing() {
  document.getElementById("analysingState").classList.remove("show");
  if (analysingIntervalId) {
    clearInterval(analysingIntervalId);
    analysingIntervalId = null;
  }
}

function renderSpeakers(speakers) {
  const section = document.getElementById("speakersSection");
  const cardsEl = document.getElementById("speakerCards");
  cardsEl.innerHTML = "";
  if (!speakers || speakers.length === 0) {
    section.style.display = "none";
    return;
  }
  section.style.display = "block";
  speakers.forEach((sp) => {
    const band = sp.final.band;
    const color = BAND_COLOR[band] || "var(--slate-soft)";
    const card = document.createElement("div");
    card.className = "speaker-card";
    card.innerHTML = `
      <div class="sc-top">
        <div>
          <div class="sc-label">${sp.speaker_label}</div>
          <div class="sc-meta">${(sp.total_duration_ms / 1000).toFixed(1)}s voiced · ${sp.segment_count} segment${sp.segment_count === 1 ? "" : "s"}</div>
        </div>
        <span class="sc-score mono" style="color:${color}">${sp.final.smoothed_score_0_100.toFixed(0)} <span style="font-size:11px; font-weight:600;">${band}</span></span>
      </div>
      <button class="sc-replay">▸ view this speaker's full breakdown</button>
    `;
    card.querySelector(".sc-replay").addEventListener("click", () => {
      resetResultsKeepSpeakers();
      sp.trace.forEach(renderFusedScore);
      renderSpeakers(speakers); // re-show cards since resetResultsKeepSpeakers cleared the section
      document.getElementById("results").scrollIntoView({ behavior: "smooth", block: "start" });
    });
    cardsEl.appendChild(card);
  });
}

function resetResultsKeepSpeakers() {
  traceScores = [];
  traceComponents = {};
  traceTimestamps = [];
  traceSignal = [];
  document.getElementById("resultsEmpty").style.display = "none";
}

function renderFusedScore(fs) {
  document.getElementById("resultsEmpty").style.display = "none";
  document.getElementById("results").classList.add("show");

  const score = fs.smoothed_score_0_100;
  const band = fs.band;
  const color = BAND_COLOR[band] || "var(--slate-soft)";

  document.getElementById("gaugeScore").textContent = score.toFixed(0);
  document.getElementById("gaugeScore").style.color = color;
  document.getElementById("gaugeBand").textContent = band.toUpperCase();
  document.getElementById("gaugeBand").style.color = color;
  const arc = document.getElementById("gaugeArc");
  arc.style.stroke = color;
  arc.setAttribute("stroke-dashoffset", String(GAUGE_CIRCUMFERENCE * (1 - Math.min(score, 100) / 100)));

  const actionBand = document.getElementById("actionBand");
  actionBand.innerHTML = "";
  actionBand.appendChild(document.createTextNode(band));
  if (BAND_HELP[band]) actionBand.appendChild(helpIcon(BAND_HELP[band]));
  actionBand.className = "action-band " + band;
  document.getElementById("actionText").textContent = fs.recommended_action;

  // transcript_source/transcript_language — added 2026-09-10 so "which
  // transcriber handled this call" is visible here, not just in logs
  // (see FusedScore's own docstring for the full account).
  const sourceNote = document.getElementById("transcriptSourceNote");
  if (fs.transcript_source) {
    const label = TRANSCRIBER_LABEL[fs.transcript_source] || fs.transcript_source;
    sourceNote.textContent = `Transcription: ${label}${fs.transcript_language ? ` (detected: ${fs.transcript_language})` : ""}`;
    sourceNote.style.display = "";
  } else {
    sourceNote.style.display = "none";
  }

  const compEl = document.getElementById("components");
  compEl.innerHTML = "";
  fs.components.forEach((c) => {
    // 2026-09-10: the primary displayed number is now smoothed_score — the
    // per-component EMA aggregate that ACTUALLY fed the fusion formula
    // (see WeightedSumFusion's own docstring) — not raw_score (this
    // window alone). A component can show a real number here even while
    // `abstained` is true for this specific window, as long as it has
    // carried-forward history; only a component that has NEVER produced a
    // score this session shows the flat "abstained" state.
    const hasValue = c.smoothed_score !== null && c.smoothed_score !== undefined;
    const pct = hasValue ? Math.round(c.smoothed_score * 100) : 0;
    const row = document.createElement("div");
    row.className = "component-row" + (!hasValue ? " abstained" : "");
    row.innerHTML = `
      <div class="top-line">
        <span class="name">${COMPONENT_LABEL[c.name] || c.name}</span>
        <span class="value mono">${hasValue ? pct + "%" : "abstained"}</span>
      </div>
      <div class="bar-track"><div class="bar-fill" style="width:${hasValue ? pct : 100}%"></div></div>
      ${c.abstained
        ? `<div class="abstain-note">${hasValue ? "carried forward from an earlier window — " : ""}${c.detail && c.detail.abstain_reason ? c.detail.abstain_reason : "no signal for this window"}</div>`
        : renderExplanationNote(c.detail)}
      ${renderTranscriptNote(c.detail)}
      ${renderIntentBreakdown(c.detail)}
    `;
    if (COMPONENT_HELP[c.name]) row.querySelector(".name").appendChild(helpIcon(COMPONENT_HELP[c.name]));
    compEl.appendChild(row);
  });

  traceScores.push(score);
  traceTimestamps.push(fs.window_start_ms);
  traceSignal.push(extractSignalIntensityPct(fs.components));
  if (traceScores.length > 80) { traceScores.shift(); traceTimestamps.shift(); traceSignal.shift(); } // cap for live mode so the chart stays readable
  fs.components.forEach((c) => {
    if (!(c.name in traceComponents)) traceComponents[c.name] = [];
    if (!(c.name in chartFactorEnabled)) chartFactorEnabled[c.name] = true; // default on, first time this factor is seen
    const arr = traceComponents[c.name];
    arr.push(c.smoothed_score === null || c.smoothed_score === undefined ? null : c.smoothed_score * 100);
    if (arr.length > 80) arr.shift(); // same rolling cap as traceScores, kept index-aligned
  });
  renderChartLegend();
  drawTraceChart();

  document.getElementById("rawJson").textContent = JSON.stringify(fs, null, 2);
}

// One chip per factor seen so far, in first-seen (== config) order.
// Clicking toggles that factor's line on/off and redraws immediately.
function renderChartLegend() {
  const el = document.getElementById("chartLegend");
  el.innerHTML = "";
  const overallChip = document.createElement("button");
  overallChip.className = "chip" + (chartFactorEnabled["__overall__"] === false ? " off" : "");
  overallChip.innerHTML = `<span class="dot" style="background:var(--ink)"></span>Overall`;
  overallChip.addEventListener("click", () => {
    chartFactorEnabled["__overall__"] = chartFactorEnabled["__overall__"] === false ? true : false;
    renderChartLegend();
    drawTraceChart();
  });
  el.appendChild(overallChip);

  const signalChip = document.createElement("button");
  signalChip.className = "chip" + (chartFactorEnabled["__signal__"] === false ? " off" : "");
  signalChip.innerHTML = `<span class="dot" style="background:${SIGNAL_CHART_COLOR}"></span>Signal intensity`;
  signalChip.addEventListener("click", () => {
    chartFactorEnabled["__signal__"] = chartFactorEnabled["__signal__"] === false ? true : false;
    renderChartLegend();
    drawTraceChart();
  });
  el.appendChild(signalChip);

  Object.keys(traceComponents).forEach((name) => {
    const chip = document.createElement("button");
    chip.className = "chip" + (chartFactorEnabled[name] ? "" : " off");
    chip.innerHTML = `<span class="dot" style="background:${componentChartColor(name)}"></span>${COMPONENT_LABEL[name] || name}`;
    chip.addEventListener("click", () => {
      chartFactorEnabled[name] = !chartFactorEnabled[name];
      renderChartLegend();
      drawTraceChart();
    });
    el.appendChild(chip);
  });
}

// Splits a (number|null)[] series into contiguous non-null runs, each
// rendered as its own <polyline> — a gap (a factor with no value at that
// point, e.g. never scored yet) is a genuine visual break, not
// interpolated across, so the chart never implies a reading that was
// never actually produced.
function seriesToPolylines(series, stepX, h, pad, leftPad) {
  const runs = [];
  let current = [];
  series.forEach((v, i) => {
    if (v === null || v === undefined) {
      if (current.length) runs.push(current);
      current = [];
      return;
    }
    const x = leftPad + i * stepX;
    const y = h - pad - (Math.min(Math.max(v, 0), 100) / 100) * (h - pad * 2);
    current.push(`${x.toFixed(1)},${y.toFixed(1)}`);
  });
  if (current.length) runs.push(current);
  return runs;
}

// Chart layout constants — shared between drawTraceChart() (which lays
// out the SVG) and the hover handlers below (which must map a mouse
// pixel position back to the same coordinate space) so the two never
// drift out of sync.
const CHART_W = 600, CHART_H = 140;
const CHART_TOP_PAD = 8, CHART_BOTTOM_PAD = 22, CHART_LEFT_PAD = 34, CHART_RIGHT_PAD = 8;
const CHART_PLOT_H = CHART_H - CHART_TOP_PAD - CHART_BOTTOM_PAD;

function chartStepX() {
  return (CHART_W - CHART_LEFT_PAD - CHART_RIGHT_PAD) / Math.max(traceScores.length - 1, 1);
}

function formatSeconds(ms) {
  const s = ms / 1000;
  return (s < 10 ? s.toFixed(1) : Math.round(s)) + "s";
}

function drawTraceChart() {
  const svg = document.getElementById("traceChart");
  if (traceScores.length < 2) { svg.innerHTML = ""; hideHover(); return; }
  const stepX = chartStepX();
  const h = CHART_H, pad = CHART_TOP_PAD, leftPad = CHART_LEFT_PAD, rightEdge = CHART_W - CHART_RIGHT_PAD;

  let linesHtml = "";
  // Signal intensity drawn FIRST (bottom-most, behind everything else) —
  // it's a diagnostic reference, not a risk factor competing for
  // attention with the score lines above it.
  if (chartFactorEnabled["__signal__"] !== false) {
    seriesToPolylines(traceSignal, stepX, h, pad, leftPad).forEach((run) => {
      linesHtml += `<polyline points="${run.join(" ")}" fill="none" stroke="${SIGNAL_CHART_COLOR}" stroke-width="1.5" stroke-dasharray="4 3" stroke-linejoin="round" stroke-linecap="round" opacity="0.7"/>`;
    });
  }
  Object.keys(traceComponents).forEach((name) => {
    if (!chartFactorEnabled[name]) return;
    seriesToPolylines(traceComponents[name], stepX, h, pad, leftPad).forEach((run) => {
      linesHtml += `<polyline points="${run.join(" ")}" fill="none" stroke="${componentChartColor(name)}" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round" opacity="0.85"/>`;
    });
  });
  // Overall drawn LAST (on top of the per-factor lines) and noticeably
  // thicker, so it reads as the "headline" line the others sit alongside.
  if (chartFactorEnabled["__overall__"] !== false) {
    const last = traceScores[traceScores.length - 1];
    const overallColor = last <= 34 ? "var(--risk-low)" : last <= 69 ? "var(--risk-med)" : "var(--risk-high)";
    seriesToPolylines(traceScores, stepX, h, pad, leftPad).forEach((run) => {
      linesHtml += `<polyline points="${run.join(" ")}" fill="none" stroke="${overallColor}" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>`;
    });
  }

  // Y-axis: 0%/50%/100% gridlines + labels, plus the existing low/elevated
  // band-boundary reference lines (34%, 69%) kept as lighter dashed lines.
  let axisHtml = "";
  [0, 50, 100].forEach((pct) => {
    const y = h - pad - (pct / 100) * (h - pad * 2);
    axisHtml += `<line x1="${leftPad}" y1="${y.toFixed(1)}" x2="${rightEdge}" y2="${y.toFixed(1)}" stroke="var(--line-soft)" stroke-width="1"/>`;
    axisHtml += `<text x="${leftPad - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end">${pct}%</text>`;
  });
  [34, 69].forEach((pct) => {
    const y = h - pad - (pct / 100) * (h - pad * 2);
    axisHtml += `<line x1="${leftPad}" y1="${y.toFixed(1)}" x2="${rightEdge}" y2="${y.toFixed(1)}" stroke="var(--line-soft)" stroke-width="1" stroke-dasharray="3 3"/>`;
  });
  // X-axis: a handful of evenly-spaced time labels (window_start_ms -> s).
  const nLabels = Math.min(5, traceTimestamps.length);
  for (let i = 0; i < nLabels; i++) {
    const idx = Math.round((i / Math.max(nLabels - 1, 1)) * (traceTimestamps.length - 1));
    const x = leftPad + idx * stepX;
    axisHtml += `<text x="${x.toFixed(1)}" y="${h - 6}" text-anchor="middle">${formatSeconds(traceTimestamps[idx])}</text>`;
  }

  svg.innerHTML = `
    ${axisHtml}
    ${linesHtml}
    <rect id="hoverCapture" x="${leftPad}" y="${pad}" width="${rightEdge - leftPad}" height="${h - pad - CHART_BOTTOM_PAD}" fill="transparent" style="cursor:crosshair"/>
    <g id="hoverLayer"></g>
  `;
}

// --- hover tooltip: attached ONCE to the outer <svg> element (which
// persists across drawTraceChart()'s innerHTML rewrites of its
// children), reading whatever traceScores/traceComponents/traceTimestamps
// currently hold — no need to re-attach on every redraw. ---------------
function nearestIndexForX(svgX) {
  if (traceScores.length < 2) return null;
  const stepX = chartStepX();
  const idx = Math.round((svgX - CHART_LEFT_PAD) / stepX);
  return idx >= 0 && idx < traceScores.length ? idx : null;
}

function hideHover() {
  const layer = document.getElementById("hoverLayer");
  if (layer) layer.innerHTML = "";
  document.getElementById("chartTooltip").style.display = "none";
}

function showHoverAt(idx, clientX, clientY) {
  const svg = document.getElementById("traceChart");
  const wrap = svg.closest(".chart-wrap");
  const layer = document.getElementById("hoverLayer");
  if (!layer) return;
  const stepX = chartStepX();
  const x = CHART_LEFT_PAD + idx * stepX;
  const yFor = (v) => CHART_H - CHART_TOP_PAD - (Math.min(Math.max(v, 0), 100) / 100) * (CHART_H - CHART_TOP_PAD * 2);

  let dotsHtml = `<line x1="${x.toFixed(1)}" y1="${CHART_TOP_PAD}" x2="${x.toFixed(1)}" y2="${CHART_H - CHART_BOTTOM_PAD}" stroke="var(--slate-soft)" stroke-width="1" stroke-dasharray="2 2"/>`;
  let rows = "";
  if (chartFactorEnabled["__overall__"] !== false) {
    const v = traceScores[idx];
    dotsHtml += `<circle cx="${x.toFixed(1)}" cy="${yFor(v).toFixed(1)}" r="3.5" fill="var(--ink)"/>`;
    rows += `<div class="tt-row"><span class="dot" style="background:var(--ink)"></span>Overall: ${Math.round(v)}%</div>`;
  }
  Object.keys(traceComponents).forEach((name) => {
    if (!chartFactorEnabled[name]) return;
    const v = traceComponents[name][idx];
    if (v === null || v === undefined) return;
    dotsHtml += `<circle cx="${x.toFixed(1)}" cy="${yFor(v).toFixed(1)}" r="3" fill="${componentChartColor(name)}"/>`;
    rows += `<div class="tt-row"><span class="dot" style="background:${componentChartColor(name)}"></span>${COMPONENT_LABEL[name] || name}: ${Math.round(v)}%</div>`;
  });
  if (chartFactorEnabled["__signal__"] !== false) {
    const v = traceSignal[idx];
    if (v !== null && v !== undefined) {
      dotsHtml += `<circle cx="${x.toFixed(1)}" cy="${yFor(v).toFixed(1)}" r="3" fill="${SIGNAL_CHART_COLOR}"/>`;
      rows += `<div class="tt-row"><span class="dot" style="background:${SIGNAL_CHART_COLOR}"></span>Signal intensity: ${Math.round(v)}%</div>`;
    }
  }
  layer.innerHTML = dotsHtml;

  const tooltip = document.getElementById("chartTooltip");
  tooltip.innerHTML = `<div class="tt-time">${formatSeconds(traceTimestamps[idx])}</div>${rows}`;
  tooltip.style.display = "block";
  const wrapRect = wrap.getBoundingClientRect();
  let left = clientX - wrapRect.left + 14;
  if (left + tooltip.offsetWidth > wrapRect.width) left = clientX - wrapRect.left - tooltip.offsetWidth - 14;
  tooltip.style.left = left + "px";
  tooltip.style.top = (clientY - wrapRect.top - tooltip.offsetHeight / 2) + "px";
}

function initChartHover() {
  const svg = document.getElementById("traceChart");
  svg.addEventListener("mousemove", (e) => {
    const capture = document.getElementById("hoverCapture");
    if (!capture) return;
    const rect = svg.getBoundingClientRect();
    const svgX = ((e.clientX - rect.left) / rect.width) * CHART_W;
    const idx = nearestIndexForX(svgX);
    if (idx === null) { hideHover(); return; }
    showHoverAt(idx, e.clientX, e.clientY);
  });
  svg.addEventListener("mouseleave", hideHover);
}
initChartHover();

document.getElementById("rawToggle").addEventListener("click", (e) => {
  const el = document.getElementById("rawJson");
  el.classList.toggle("show");
  e.target.textContent = (el.classList.contains("show") ? "▾" : "▸") + " view raw JSON";
});

function showError(boxId, message) {
  const box = document.getElementById(boxId);
  box.textContent = message;
  box.classList.add("show");
}
function clearError(boxId) {
  document.getElementById(boxId).classList.remove("show");
}

// Reads a fetch Response as JSON without assuming the SERVER actually sent
// JSON. REAL BUG this was written for: an unhandled backend exception (e.g.
// a missing model file) makes FastAPI/Starlette's default error handler
// return a plain-text 500 body ("Internal Server Error"), not JSON — code
// that did `const body = await res.json()` unconditionally then threw
// inside that parse, and the resulting SyntaxError's browser-native message
// ("The string did not match the expected pattern." in Safari; "Unexpected
// token I in JSON at position 0" in Chrome) is what actually reached the
// user, with zero indication a request even failed server-side or why.
// This always reads the body as text first, so a non-JSON error body still
// produces an actionable message instead of a parse-error message.
async function parseJsonResponse(res) {
  const text = await res.text();
  let body = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    // Not JSON — fall through with body === null; callers use res.ok /
    // res.status and the raw text below instead.
  }
  if (!res.ok) {
    const detail = body && body.detail;
    throw new Error(
      detail || (text ? text.slice(0, 300) : `request failed (HTTP ${res.status})`)
    );
  }
  if (body === null) {
    // res.ok but the body wasn't valid JSON — shouldn't happen for this
    // API's success responses, but fail with a clear message rather than
    // returning null and letting a caller crash on body.whatever instead.
    throw new Error(`expected a JSON response but got: ${text.slice(0, 300)}`);
  }
  return body;
}

// ---------- upload tab ----------
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const analyzeFileBtn = document.getElementById("analyzeFileBtn");
let selectedFile = null;

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") fileInput.click(); });
["dragenter", "dragover"].forEach((evt) => dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach((evt) => dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); }));
dropzone.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) setSelectedFile(e.dataTransfer.files[0]); });
fileInput.addEventListener("change", () => { if (fileInput.files[0]) setSelectedFile(fileInput.files[0]); });

function setSelectedFile(file) {
  selectedFile = file;
  document.getElementById("fileName").textContent = file.name;
  analyzeFileBtn.disabled = false;
  clearError("uploadError");
}

function getAudioDurationSeconds(file) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const audioEl = new Audio();
    const cleanup = () => URL.revokeObjectURL(url);
    audioEl.addEventListener("loadedmetadata", () => {
      const d = Number.isFinite(audioEl.duration) ? audioEl.duration : null;
      cleanup();
      resolve(d);
    });
    audioEl.addEventListener("error", () => { cleanup(); resolve(null); });
    setTimeout(() => { cleanup(); resolve(null); }, 2000); // don't block on an odd format
    audioEl.src = url;
  });
}

analyzeFileBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  clearError("uploadError");
  analyzeFileBtn.disabled = true;
  analyzeFileBtn.textContent = "Analysing…";
  resetResults();

  const diarizeOn = document.getElementById("diarizeCheck").checked;
  // Named from the live /v1/config detector list, not a hardcoded string, so this can't
  // silently go stale again the next time a detector is added or removed.
  const detectorNames = lastConfig && lastConfig.detectors && lastConfig.detectors.length
    ? lastConfig.detectors.map((d) => COMPONENT_LABEL[d.name] || d.name).join(", ")
    : "every configured detector";
  showAnalysing(
    `Running ${detectorNames} over every window` +
      (diarizeOn ? ", then diarizing and re-scoring per speaker." : ".") +
      " Longer recordings take longer — this runs entirely on CPU."
  );
  getAudioDurationSeconds(selectedFile).then((durationS) => {
    if (durationS && document.getElementById("analysingState").classList.contains("show")) {
      document.getElementById("analysingDetail").textContent =
        `~${durationS.toFixed(1)}s of audio. ` + document.getElementById("analysingDetail").textContent;
    }
  });

  try {
    const form = new FormData();
    form.append("file", selectedFile);
    form.append("context", JSON.stringify(readContext(document.getElementById("ctxUpload"))));
    form.append("diarize", diarizeOn ? "true" : "false");

    const res = await fetch(`${BASE_PATH}/v1/score/file`, { method: "POST", body: form });
    const body = await parseJsonResponse(res);

    hideAnalysing();
    body.trace.forEach(renderFusedScore); // replays the whole trace so the chart shows the full call
    renderSpeakers(body.speakers);
  } catch (err) {
    hideAnalysing();
    document.getElementById("resultsEmpty").style.display = "flex";
    showError("uploadError", err.message || String(err));
  } finally {
    analyzeFileBtn.disabled = false;
    analyzeFileBtn.textContent = "Analyse file";
  }
});

// ---------- microphone (live) tab ----------
const micBtn = document.getElementById("micBtn");
const micStatus = document.getElementById("micStatus");
const levelBar = document.getElementById("levelBar");

let isRecording = false;
let audioCtx, analyserNode, processorNode, mediaStream, ws;
let sampleChunks = [];
let totalSamples = 0;
let samplesSinceLastEmit = 0;
let windowSeq = 0;
let sessionStartMs = 0;
let levelRAF = null;

// Opt-in full-session recording (separate from sampleChunks above, which
// is capped/trimmed for the rolling-window use case — this one is never
// trimmed, only gated by the checkbox, and exists purely so the user can
// download what was actually captured). See encodeWav() below.
let recordFullSession = false;
let recordedChunks = [];
const recordCheckbox = document.getElementById("recordMicCheckbox");
const micRecordingDownload = document.getElementById("micRecordingDownload");

const WINDOW_MS = 2000, HOP_MS = 500; // mirrors config/risk_formula.yaml defaults

micBtn.addEventListener("click", () => { isRecording ? stopMic() : startMic(); });

async function startMic() {
  clearError("micError");
  try {
    // Explicitly OFF, not left at browser defaults (which is "on" for all
    // three in every current browser) — real bug found 2026-09-08: AASIST's
    // live-mic spoof_probability was observed oscillating wildly (93.6% ->
    // 90.8% -> 59.6% -> 72.8% -> 0.2%) across consecutive windows of
    // continuous genuine English speech, correlating with loudness.
    // autoGainControl continuously renormalises mic gain based on recent
    // loudness — a real-time dynamic-range distortion AASIST never saw
    // during training (ASVspoof2019 LA is raw studio audio, no AGC).
    // noiseSuppression applies its own real-time spectral filter, a second
    // plausible source of artifacts an anti-spoofing model is sensitive to.
    // HONESTY NOTE: these are requests, not guarantees — a browser/OS may
    // still apply some processing regardless (MediaStream constraints are
    // "best effort"). Real trade-off accepted here: a genuinely noisy room
    // now passes more raw noise into the pipeline than before, in exchange
    // for not distorting the exact acoustic properties the acoustic and
    // prosodic detectors key on.
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
  } catch (err) {
    showError("micError", "Microphone permission denied or unavailable: " + err.message);
    return;
  }

  resetResults();
  sampleChunks = [];
  totalSamples = 0;
  samplesSinceLastEmit = 0;
  windowSeq = 0;
  sessionStartMs = performance.now();

  recordFullSession = !!(recordCheckbox && recordCheckbox.checked);
  recordedChunks = [];
  if (micRecordingDownload) {
    // Release the previous session's recording (if any) before replacing it.
    if (micRecordingDownload.href) URL.revokeObjectURL(micRecordingDownload.href);
    micRecordingDownload.removeAttribute("href");
    micRecordingDownload.style.display = "none";
  }

  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioCtx.createMediaStreamSource(mediaStream);

  analyserNode = audioCtx.createAnalyser();
  analyserNode.fftSize = 512;
  source.connect(analyserNode);

  // AudioWorkletNode (migrated 2026-09-11 from the deprecated
  // ScriptProcessorNode — see mic-worklet.js's own docstring for why:
  // ScriptProcessorNode's callback runs on the main thread, so page JS
  // or a GC pause can delay/drop audio callbacks, a real source of
  // missing samples feeding live transcription, not just an API-
  // cleanliness concern). addModule() is itself async, so this can
  // fail (404, insecure context, browser without AudioWorklet support)
  // — surfaced as a real micError rather than a silent dead mic.
  try {
    await audioCtx.audioWorklet.addModule(`${BASE_PATH}/mic-worklet.js`);
  } catch (err) {
    showError("micError", "Could not load the audio capture module: " + err.message);
    mediaStream.getTracks().forEach((t) => t.stop());
    audioCtx.close();
    return;
  }
  processorNode = new AudioWorkletNode(audioCtx, "mic-capture-processor");
  source.connect(processorNode);
  processorNode.connect(audioCtx.destination); // required to keep it firing in some browsers; we never write to the output buffer, so it stays silent

  const sessionId = "live-" + Date.now();
  const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${wsProtocol}//${location.host}${BASE_PATH}/v1/stream/${sessionId}`);
  ws.onmessage = (evt) => {
    const data = JSON.parse(evt.data);
    if (data.error) { showError("micError", "server rejected a chunk: " + data.error); return; }
    renderFusedScore(data);
  };
  ws.onerror = () => showError("micError", "WebSocket connection error — is the service reachable?");

  processorNode.port.onmessage = (e) => {
    const input = e.data; // Float32Array, 4096 samples — same chunk size the old ScriptProcessorNode used
    sampleChunks.push(input);
    totalSamples += input.length;
    samplesSinceLastEmit += input.length;

    if (recordFullSession) recordedChunks.push(input); // never trimmed — see its own declaration comment

    const hopSamples = Math.round(audioCtx.sampleRate * (HOP_MS / 1000));
    const windowSamples = Math.round(audioCtx.sampleRate * (WINDOW_MS / 1000));

    if (samplesSinceLastEmit >= hopSamples && ws.readyState === WebSocket.OPEN) {
      const windowData = trailingWindow(sampleChunks, windowSamples);
      const elapsedMs = Math.round(performance.now() - sessionStartMs);
      ws.send(JSON.stringify({
        seq: windowSeq++,
        sample_rate: audioCtx.sampleRate,
        pcm_f32: Array.from(windowData),
        window_start_ms: Math.max(0, elapsedMs - WINDOW_MS),
        context: readContext(document.getElementById("ctxMic")),
      }));
      samplesSinceLastEmit = 0;
      trimOldChunks(windowSamples * 3); // cap memory for long sessions
    }
  };

  isRecording = true;
  micBtn.classList.add("recording");
  micStatus.innerHTML = 'Listening — <span class="mono">speak now</span>. Click again to stop.';
  animateLevel();
}

function trailingWindow(chunks, neededSamples) {
  const total = chunks.reduce((s, c) => s + c.length, 0);
  const outLen = Math.min(neededSamples, total);
  const out = new Float32Array(outLen);
  let filled = 0;
  for (let i = chunks.length - 1; i >= 0 && filled < outLen; i--) {
    const c = chunks[i];
    const take = Math.min(c.length, outLen - filled);
    out.set(c.subarray(c.length - take), outLen - filled - take);
    filled += take;
  }
  return out;
}

function trimOldChunks(keepSamples) {
  let total = sampleChunks.reduce((s, c) => s + c.length, 0);
  while (sampleChunks.length > 1 && total - sampleChunks[0].length > keepSamples) {
    total -= sampleChunks.shift().length;
  }
}

function animateLevel() {
  if (!isRecording || !analyserNode) return;
  const data = new Uint8Array(analyserNode.fftSize);
  analyserNode.getByteTimeDomainData(data);
  let sumSq = 0;
  for (let i = 0; i < data.length; i++) { const v = (data[i] - 128) / 128; sumSq += v * v; }
  const rms = Math.sqrt(sumSq / data.length);
  levelBar.style.width = Math.min(100, rms * 300) + "%";
  levelRAF = requestAnimationFrame(animateLevel);
}

function stopMic() {
  isRecording = false;
  micBtn.classList.remove("recording");
  micStatus.textContent = "Click to start — your browser will ask for microphone permission.";
  levelBar.style.width = "0%";
  if (levelRAF) cancelAnimationFrame(levelRAF);
  const sampleRateForRecording = audioCtx ? audioCtx.sampleRate : null;
  if (processorNode) { processorNode.disconnect(); processorNode.port.onmessage = null; }
  if (analyserNode) analyserNode.disconnect();
  if (audioCtx) audioCtx.close();
  if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
  if (ws && ws.readyState === WebSocket.OPEN) ws.close();

  if (recordFullSession && recordedChunks.length && sampleRateForRecording && micRecordingDownload) {
    const blob = encodeWav(recordedChunks, sampleRateForRecording);
    micRecordingDownload.href = URL.createObjectURL(blob);
    micRecordingDownload.style.display = "";
  }
  recordFullSession = false;
  recordedChunks = [];
}

// Encodes raw float32 PCM chunks (as captured from mic-worklet.js's
// AudioWorkletNode above) into a standard 16-bit-PCM mono WAV file —
// no server round-trip, no library: just a 44-byte RIFF/WAVE/fmt/data
// header written directly into an ArrayBuffer ahead of the sample data.
// Kept entirely client-side since this is a convenience download of
// exactly what the browser already captured, not a new server capability.
function encodeWav(float32Chunks, sampleRate) {
  const totalLen = float32Chunks.reduce((sum, c) => sum + c.length, 0);
  const pcm16 = new Int16Array(totalLen);
  let offset = 0;
  for (const chunk of float32Chunks) {
    for (let i = 0; i < chunk.length; i++) {
      const clamped = Math.max(-1, Math.min(1, chunk[i]));
      pcm16[offset++] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    }
  }

  const bytesPerSample = 2, numChannels = 1;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = pcm16.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeStr = (offset, str) => { for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i)); };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);       // fmt chunk size
  view.setUint16(20, 1, true);        // PCM format
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bytesPerSample * 8, true); // bits per sample
  writeStr(36, "data");
  view.setUint32(40, dataSize, true);

  for (let i = 0; i < pcm16.length; i++) view.setInt16(44 + i * 2, pcm16[i], true);

  return new Blob([buffer], { type: "audio/wav" });
}

// ---------- history tab ----------
document.getElementById("refreshHistoryBtn").addEventListener("click", loadHistoryList);

async function loadHistoryList() {
  clearError("historyError");
  const listEl = document.getElementById("historyList");
  listEl.innerHTML = '<div class="history-empty">Loading…</div>';
  try {
    const res = await fetch(`${BASE_PATH}/v1/sessions?limit=50`);
    if (!res.ok) throw new Error("could not load session history");
    const sessions = await res.json();

    if (sessions.length === 0) {
      listEl.innerHTML = '<div class="history-empty">No sessions yet — upload a file or use the microphone, then check back here.</div>';
      return;
    }

    listEl.innerHTML = "";
    sessions.forEach((s) => listEl.appendChild(buildHistoryRow(s)));
  } catch (err) {
    showError("historyError", err.message || String(err));
    listEl.innerHTML = "";
  }
}

function formatSessionLabel(sessionId) {
  // Per-speaker sub-sessions (see docs/architecture.md, "Diarization")
  // are named "{base_session_id}::speaker_N" — clear to a developer, not
  // to someone reading the History tab. Show "Speaker N" as the primary
  // label with the underlying call id de-emphasised, rather than the raw
  // id as-is. Plain (non-diarized) session ids are shown unchanged.
  const speakerMatch = sessionId.match(/^(.+)::(speaker_\d+)$/);
  if (!speakerMatch) return escapeHtml(sessionId);
  const [, baseId, speakerPart] = speakerMatch;
  const speakerNum = speakerPart.replace("speaker_", "");
  const shortBase = baseId.length > 20 ? baseId.slice(0, 20) + "…" : baseId;
  return `<strong>Speaker ${speakerNum}</strong> <span class="history-id-sub">— from call ${escapeHtml(shortBase)}</span>`;
}

function buildHistoryRow(s) {
  const row = document.createElement("div");
  row.className = "history-row";
  const started = new Date(s.started_at).toLocaleString();
  row.innerHTML = `
    <span class="history-band-dot ${s.final_band}"></span>
    <div class="history-main">
      <div class="history-id">${formatSessionLabel(s.session_id)}</div>
      <div class="history-meta">${started} · ${s.window_count} window${s.window_count === 1 ? "" : "s"}</div>
    </div>
    <span class="history-score mono" style="color:${BAND_COLOR[s.final_band] || "var(--slate-soft)"}">${s.final_smoothed_score.toFixed(0)}</span>
    <button class="history-delete" title="Delete this session">✕</button>
  `;
  row.querySelector(".history-main").addEventListener("click", () => openHistorySession(s.session_id));
  row.querySelector(".history-score").addEventListener("click", () => openHistorySession(s.session_id));
  row.querySelector(".history-delete").addEventListener("click", async (e) => {
    e.stopPropagation();
    if (!confirm(`Delete session ${s.session_id}? This can't be undone.`)) return;
    await fetch(`${BASE_PATH}/v1/sessions/${encodeURIComponent(s.session_id)}`, { method: "DELETE" });
    loadHistoryList();
  });
  return row;
}

async function openHistorySession(sessionId) {
  clearError("historyError");
  try {
    const res = await fetch(`${BASE_PATH}/v1/sessions/${encodeURIComponent(sessionId)}`);
    if (!res.ok) throw new Error("could not load that session — it may have been deleted");
    const body = await res.json();
    resetResults();
    body.trace.forEach(renderFusedScore); // same replay pattern as the upload tab
  } catch (err) {
    showError("historyError", err.message || String(err));
  }
}

// ---------- voiceprints (enroll) tab ----------
const enrollDropzone = document.getElementById("enrollDropzone");
const enrollFileInput = document.getElementById("enrollFileInput");
const enrollBtn = document.getElementById("enrollBtn");
const enrollIdentityInput = document.getElementById("enrollIdentity");
let selectedEnrollFile = null;

enrollDropzone.addEventListener("click", () => enrollFileInput.click());
enrollDropzone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") enrollFileInput.click(); });
["dragenter", "dragover"].forEach((evt) => enrollDropzone.addEventListener(evt, (e) => { e.preventDefault(); enrollDropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach((evt) => enrollDropzone.addEventListener(evt, (e) => { e.preventDefault(); enrollDropzone.classList.remove("drag"); }));
enrollDropzone.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) setSelectedEnrollFile(e.dataTransfer.files[0]); });
enrollFileInput.addEventListener("change", () => { if (enrollFileInput.files[0]) setSelectedEnrollFile(enrollFileInput.files[0]); });

function setSelectedEnrollFile(file) {
  selectedEnrollFile = file;
  document.getElementById("enrollFileName").textContent = file.name;
  updateEnrollBtnState();
  clearError("enrollError");
}
enrollIdentityInput.addEventListener("input", updateEnrollBtnState);
function updateEnrollBtnState() {
  enrollBtn.disabled = !(selectedEnrollFile && enrollIdentityInput.value.trim());
}

enrollBtn.addEventListener("click", async () => {
  const identity = enrollIdentityInput.value.trim();
  if (!selectedEnrollFile || !identity) return;
  clearError("enrollError");
  enrollBtn.disabled = true;
  enrollBtn.textContent = "Enrolling…";
  try {
    const form = new FormData();
    form.append("identity", identity);
    form.append("file", selectedEnrollFile);
    const res = await fetch(`${BASE_PATH}/v1/enroll`, { method: "POST", body: form });
    await parseJsonResponse(res);
    selectedEnrollFile = null;
    document.getElementById("enrollFileName").textContent = "";
    enrollIdentityInput.value = "";
    loadEnrollmentList();
  } catch (err) {
    showError("enrollError", err.message || String(err));
  } finally {
    enrollBtn.textContent = "Enroll voiceprint";
    updateEnrollBtnState();
  }
});

document.getElementById("refreshEnrollmentsBtn").addEventListener("click", loadEnrollmentList);

async function loadEnrollmentList() {
  const listEl = document.getElementById("enrollmentList");
  listEl.innerHTML = '<div class="history-empty">Loading…</div>';
  try {
    const res = await fetch(`${BASE_PATH}/v1/enrollments`);
    if (!res.ok) throw new Error("could not load enrollments (voiceprint consistency detector may not be configured)");
    const enrollments = await res.json();

    if (enrollments.length === 0) {
      listEl.innerHTML = '<div class="history-empty">No voiceprints enrolled yet.</div>';
      return;
    }

    listEl.innerHTML = "";
    enrollments.forEach((e) => listEl.appendChild(buildEnrollmentRow(e)));
  } catch (err) {
    listEl.innerHTML = `<div class="history-empty">${err.message || String(err)}</div>`;
  }
}

function buildEnrollmentRow(e) {
  const row = document.createElement("div");
  row.className = "history-row";
  const updated = new Date(e.updated_at).toLocaleString();
  row.innerHTML = `
    <div class="history-main">
      <div class="history-id">${e.identity}</div>
      <div class="history-meta">enrolled ${updated} · ${e.embedding_model}</div>
    </div>
    <button class="history-delete" title="Delete this voiceprint">✕</button>
  `;
  row.querySelector(".history-delete").addEventListener("click", async () => {
    if (!confirm(`Delete the voiceprint enrolled for "${e.identity}"? This can't be undone.`)) return;
    await fetch(`${BASE_PATH}/v1/enrollments/${encodeURIComponent(e.identity)}`, { method: "DELETE" });
    loadEnrollmentList();
  });
  return row;
}
