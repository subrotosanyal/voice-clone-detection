// mic-worklet.js — AudioWorkletProcessor replacing the deprecated
// ScriptProcessorNode app.js's startMic() used to use (migrated
// 2026-09-11: ScriptProcessorNode runs its callback on the MAIN thread,
// so a busy main thread (page JS, GC pause, another tab's tab-switch
// jank) can delay or drop audio callbacks — a real, not just
// theoretical, source of missing/glitched samples feeding live
// transcription, on top of ScriptProcessorNode being formally
// deprecated. AudioWorkletNode's process() runs on the dedicated
// real-time audio rendering thread instead, isolated from main-thread
// contention.
//
// Buffers incoming 128-frame render quanta (the Web Audio spec's fixed
// callback size) up to CHUNK_SAMPLES, then transfers one Float32Array to
// the main thread — same 4096-sample chunk size the old
// ScriptProcessorNode used, so app.js's own windowing/hop logic
// (WINDOW_MS/HOP_MS in startMic()) needed no change downstream from this
// migration, only how the raw samples arrive.
const CHUNK_SAMPLES = 4096;

class MicCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = new Float32Array(CHUNK_SAMPLES);
    this._filled = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true; // no input yet (e.g. track still spinning up) — keep the node alive

    let offset = 0;
    while (offset < channel.length) {
      const space = this._buffer.length - this._filled;
      const take = Math.min(space, channel.length - offset);
      this._buffer.set(channel.subarray(offset, offset + take), this._filled);
      this._filled += take;
      offset += take;

      if (this._filled === this._buffer.length) {
        // Transfer the underlying ArrayBuffer (zero-copy across the
        // worklet/main-thread boundary) rather than cloning it, then
        // allocate a fresh buffer for the next chunk.
        this.port.postMessage(this._buffer, [this._buffer.buffer]);
        this._buffer = new Float32Array(CHUNK_SAMPLES);
        this._filled = 0;
      }
    }
    return true; // keep the processor alive for the life of the node
  }
}

registerProcessor("mic-capture-processor", MicCaptureProcessor);
