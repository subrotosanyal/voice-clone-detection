// Satya-Vani — Live Call Path — browser dashboard.
// No build step, no framework: this talks to the same two endpoints a
// curl command would (POST /v1/score/file and WS /v1/stream/{session_id}),
// documented in docs/architecture.md. If you're reading this to understand
// the wire format, those two routes (app/api/http_router.py,
// app/api/ws_router.py) are the source of truth.

const BAND_COLOR = { low: "var(--risk-low)", elevated: "var(--risk-med)", high: "var(--risk-high)" };
const COMPONENT_LABEL = { acoustic: "Acoustic (AASIST)", prosodic: "Prosodic", third_signal: "Third signal" };
const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 62;

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
  });
});

// ---------- context fields (shared builder for both tabs) ----------
function buildContextFields(container) {
  container.innerHTML = `
    <div class="field checkbox"><input type="checkbox" class="f-known"><label>Caller's number is known / on file</label></div>
    <div class="field checkbox"><input type="checkbox" class="f-financial"><label>A financial action is being requested</label></div>
    <div class="field">
      <label>Hour of day</label>
      <select class="f-hour"></select>
    </div>
    <div class="field">
      <label>Urgency language heard (comma-separated)</label>
      <input type="text" class="f-urgency" placeholder="e.g. immediately, don't tell anyone">
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
  const hour = parseInt(scopeEl.querySelector(".f-hour").value, 10);
  const urgencyRaw = scopeEl.querySelector(".f-urgency").value.trim();
  const urgency_keywords = urgencyRaw ? urgencyRaw.split(",").map((s) => s.trim()).filter(Boolean) : [];
  return { known_number: known, is_financial_request: financial, hour_of_day: hour, urgency_keywords };
}

// ---------- results rendering (shared by both modes) ----------
function resetResults() {
  traceScores = [];
  document.getElementById("resultsEmpty").style.display = "flex";
  document.getElementById("results").classList.remove("show");
}

function renderFusedScore(fs) {
  document.getElementById("resultsEmpty").style.display = "none";
  document.getElementById("results").classList.add("show");

  const score = fs.smoothed_score_0_100;
  const band = fs.band;
  const color = BAND_COLOR[band] || "var(--slate-soft)";

  document.getElementById("gaugeScore").textContent = score.toFixed(0);
  document.getElementById("gaugeScore").style.color = color;
  document.getElementById("gaugeBand").textContent = band;
  document.getElementById("gaugeBand").style.color = color;
  const arc = document.getElementById("gaugeArc");
  arc.style.stroke = color;
  arc.setAttribute("stroke-dashoffset", String(GAUGE_CIRCUMFERENCE * (1 - Math.min(score, 100) / 100)));

  const actionBand = document.getElementById("actionBand");
  actionBand.textContent = band;
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
      ${c.abstained ? `<div class="abstain-note">${c.detail && c.detail.abstain_reason ? c.detail.abstain_reason : "no signal for this window"}</div>` : ""}
    `;
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

analyzeFileBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  clearError("uploadError");
  analyzeFileBtn.disabled = true;
  analyzeFileBtn.textContent = "Analysing…";
  resetResults();

  try {
    const form = new FormData();
    form.append("file", selectedFile);
    form.append("context", JSON.stringify(readContext(document.getElementById("ctxUpload"))));

    const res = await fetch("/v1/score/file", { method: "POST", body: form });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || "analysis failed");

    body.trace.forEach(renderFusedScore); // replays the whole trace so the chart shows the full call
  } catch (err) {
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
