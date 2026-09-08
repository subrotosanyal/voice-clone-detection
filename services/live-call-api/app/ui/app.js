// Satya-Vani — Live Call Path — browser dashboard.
// No build step, no framework: this talks to the same two endpoints a
// curl command would (POST /v1/score/file and WS /v1/stream/{session_id}),
// documented in docs/architecture.md. If you're reading this to understand
// the wire format, those two routes (app/api/http_router.py,
// app/api/ws_router.py) are the source of truth.

const BAND_COLOR = { low: "var(--risk-low)", elevated: "var(--risk-med)", high: "var(--risk-high)" };
const COMPONENT_LABEL = {
  acoustic: "Acoustic (AASIST)",
  prosodic: "Prosodic",
  third_signal: "Third signal",
  intent: "Intent (zero-shot)",
};
const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 62;

// ---------- help tooltips — plain-language explanations for jargon in the UI ----------
const COMPONENT_HELP = {
  acoustic: "Detects whether the VOICE ITSELF was AI-generated or cloned, as opposed to a real person speaking — using AASIST, a neural network trained specifically to spot the audio artifacts text-to-speech and voice-cloning tools leave behind that a real human voice doesn't have. Higher % = more likely synthetic/cloned. Only validated on English speech so far.",
  prosodic: "Detects whether the voice's pitch and loudness are suspiciously steady — a real human voice naturally wavers a little from breath to breath (jitter, shimmer); a voice that's unusually 'too smooth' can be a sign of synthesis. Higher % = less natural variation than typical speech. This is a heuristic rule of thumb, not a trained classifier like Acoustic.",
  third_signal: "A third, swappable check — NOT about the audio itself. Either (a) red flags about the CALL: an unknown number, an odd hour, urgent/pressuring language, an authority claim (bank/police/government) — especially combined with a financial request, the classic fraud script — or (b) a direct voice match check against a caller's enrolled voiceprint (Voiceprints tab), when one exists. Whichever ran is named in 'view raw JSON'.",
  intent: "Scores the TRANSCRIPT (not the audio) against fraud-relevant candidate labels — 'requesting a money transfer', 'requesting an OTP/PIN', 'impersonating a bank or government official', 'creating urgency', or 'ordinary conversation' — using a zero-shot language model, not exact keyword matching like Third signal. KNOWN LIMITATION: testing found it sometimes misjudges completely ordinary conversation as suspicious — the breakdown below shows every candidate's own score, not just the winner, specifically so you can sanity-check it rather than trust one number blindly. Weighted low in the overall score for that reason. Abstains with no transcript.",
};
const BAND_HELP = {
  low: "LOW risk (score 0–34): nothing here looks suspicious across the signals that ran. Recommended action: no special handling needed.",
  elevated: "ELEVATED risk (score 35–69): at least one signal flagged something worth a closer look. Recommended action: verify the caller further before acting on a sensitive request.",
  high: "HIGH risk (score 70–100): multiple signals agree something is off, or one signal is very confident. Recommended action: block or escalate the sensitive request for manual review.",
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
  return `
    <div class="transcript-note">
      <div class="transcript-label">Transcript (auto-detected, feeds this signal)</div>
      <div class="transcript-text">"${escapeHtml(detail.transcript)}"</div>
      ${keywords.length ? `<div class="transcript-tags">urgency: ${keywords.map((k) => `<span class="tag">${escapeHtml(k)}</span>`).join(" ")}</div>` : ""}
      ${authorityKeywords.length ? `<div class="transcript-tags">authority claim: ${authorityKeywords.map((k) => `<span class="tag">${escapeHtml(k)}</span>`).join(" ")}</div>` : ""}
      ${financial ? `<div class="transcript-tags"><span class="tag owner">financial request detected</span></div>` : ""}
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

let traceScores = []; // rolling [0-100] scores for the current session/analysis, drives the chart

// ---------- health chip ----------
async function checkHealth() {
  const chip = document.getElementById("healthChip");
  const text = document.getElementById("healthText");
  try {
    const res = await fetch("/healthz");
    if (!res.ok) throw new Error("not ok");
    chip.classList.add("ok");
    text.textContent = "service healthy";
    const cfg = await (await fetch("/v1/config")).json();
    document.getElementById("formulaVersion").textContent = cfg.formula_version;
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

  const compEl = document.getElementById("components");
  compEl.innerHTML = "";
  fs.components.forEach((c) => {
    const row = document.createElement("div");
    row.className = "component-row" + (c.abstained ? " abstained" : "");
    const pct = c.abstained ? 0 : Math.round((c.raw_score || 0) * 100);
    row.innerHTML = `
      <div class="top-line">
        <span class="name">${COMPONENT_LABEL[c.name] || c.name}</span>
        <span class="value mono">${c.abstained ? "abstained" : pct + "%"}</span>
      </div>
      <div class="bar-track"><div class="bar-fill" style="width:${c.abstained ? 100 : pct}%"></div></div>
      ${c.abstained ? `<div class="abstain-note">${c.detail && c.detail.abstain_reason ? c.detail.abstain_reason : "no signal for this window"}</div>` : renderExplanationNote(c.detail)}
      ${renderTranscriptNote(c.detail)}
      ${renderIntentBreakdown(c.detail)}
    `;
    if (COMPONENT_HELP[c.name]) row.querySelector(".name").appendChild(helpIcon(COMPONENT_HELP[c.name]));
    compEl.appendChild(row);
  });

  traceScores.push(score);
  if (traceScores.length > 80) traceScores.shift(); // cap for live mode so the chart stays readable
  drawTraceChart();

  document.getElementById("rawJson").textContent = JSON.stringify(fs, null, 2);
}

function drawTraceChart() {
  const svg = document.getElementById("traceChart");
  const w = 560, h = 110, pad = 6;
  if (traceScores.length < 2) { svg.innerHTML = ""; return; }
  const stepX = (w - pad * 2) / (traceScores.length - 1);
  const points = traceScores.map((s, i) => {
    const x = pad + i * stepX;
    const y = h - pad - (Math.min(s, 100) / 100) * (h - pad * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const last = traceScores[traceScores.length - 1];
  const color = last <= 34 ? "var(--risk-low)" : last <= 69 ? "var(--risk-med)" : "var(--risk-high)";
  svg.innerHTML = `
    <line x1="${pad}" y1="${h - pad - (h - pad * 2) * 0.34}" x2="${w - pad}" y2="${h - pad - (h - pad * 2) * 0.34}" stroke="var(--line-soft)" stroke-width="1" stroke-dasharray="3 3"/>
    <line x1="${pad}" y1="${h - pad - (h - pad * 2) * 0.69}" x2="${w - pad}" y2="${h - pad - (h - pad * 2) * 0.69}" stroke="var(--line-soft)" stroke-width="1" stroke-dasharray="3 3"/>
    <polyline points="${points.join(" ")}" fill="none" stroke="${color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
  `;
}

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
  showAnalysing(
    "Running AASIST, Praat, and the third signal over every window" +
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

    const res = await fetch("/v1/score/file", { method: "POST", body: form });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || "analysis failed");

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

const WINDOW_MS = 2000, HOP_MS = 500; // mirrors config/risk_formula.yaml defaults

micBtn.addEventListener("click", () => { isRecording ? stopMic() : startMic(); });

async function startMic() {
  clearError("micError");
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
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

  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioCtx.createMediaStreamSource(mediaStream);

  analyserNode = audioCtx.createAnalyser();
  analyserNode.fftSize = 512;
  source.connect(analyserNode);

  // ScriptProcessorNode is deprecated in favour of AudioWorkletNode, but it
  // needs no separate module file to load and works in every current
  // browser — a reasonable trade-off for an internal demo tool. Swap for
  // AudioWorkletNode if this is ever hardened for production.
  processorNode = audioCtx.createScriptProcessor(4096, 1, 1);
  source.connect(processorNode);
  processorNode.connect(audioCtx.destination); // required to keep it firing in some browsers; we never write to the output buffer, so it stays silent

  const sessionId = "live-" + Date.now();
  const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${wsProtocol}//${location.host}/v1/stream/${sessionId}`);
  ws.onmessage = (evt) => {
    const data = JSON.parse(evt.data);
    if (data.error) { showError("micError", "server rejected a chunk: " + data.error); return; }
    renderFusedScore(data);
  };
  ws.onerror = () => showError("micError", "WebSocket connection error — is the service reachable?");

  processorNode.onaudioprocess = (e) => {
    const input = e.inputBuffer.getChannelData(0);
    sampleChunks.push(new Float32Array(input));
    totalSamples += input.length;
    samplesSinceLastEmit += input.length;

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
  if (processorNode) { processorNode.disconnect(); processorNode.onaudioprocess = null; }
  if (analyserNode) analyserNode.disconnect();
  if (audioCtx) audioCtx.close();
  if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
  if (ws && ws.readyState === WebSocket.OPEN) ws.close();
}

// ---------- history tab ----------
document.getElementById("refreshHistoryBtn").addEventListener("click", loadHistoryList);

async function loadHistoryList() {
  clearError("historyError");
  const listEl = document.getElementById("historyList");
  listEl.innerHTML = '<div class="history-empty">Loading…</div>';
  try {
    const res = await fetch("/v1/sessions?limit=50");
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
    await fetch(`/v1/sessions/${encodeURIComponent(s.session_id)}`, { method: "DELETE" });
    loadHistoryList();
  });
  return row;
}

async function openHistorySession(sessionId) {
  clearError("historyError");
  try {
    const res = await fetch(`/v1/sessions/${encodeURIComponent(sessionId)}`);
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
    const res = await fetch("/v1/enroll", { method: "POST", body: form });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || "enrollment failed");
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
    const res = await fetch("/v1/enrollments");
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
    await fetch(`/v1/enrollments/${encodeURIComponent(e.identity)}`, { method: "DELETE" });
    loadEnrollmentList();
  });
  return row;
}
