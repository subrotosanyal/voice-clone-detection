"""Orchestrates one window through: detectors -> fusion -> log.

This is the only place that knows about ALL of: windowing, the detector
list, the fusion strategy, and per-session smoothing state. Everything it
calls is a port; everything it produces is logged in full before it's
returned, so the log is always at least as complete as the API response —
you can debug from the logs alone, without re-running anything.

"Parallel" in the blueprint's sense means the three detectors are
independent of each other, not that this v0 necessarily runs them on
separate threads — they're small, fast, numpy-bound computations, and
running them in a plain loop keeps this trivially debuggable (a stack
trace points at exactly one detector). If a future detector is slow
(a real network-bound model call), thread/process them behind the same
DetectorPort — nothing else here needs to change.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np

from app.adapters.registry import Pipeline
from app.adapters.transcription.urgency_language import detect_urgency_signals
from app.domain.models import AudioWindow, FusedScore, SpeakerCallResult, SpeakerCallSummary
from app.logging_setup import get_logger
from app.pipeline.windowing import make_windows
from app.ports.diarizer import DiarizerPort
from app.ports.history_store import HistoryStorePort
from app.ports.intent_classifier import IntentClassifierPort
from app.ports.semantic_risk_classifier import SemanticRiskClassifierPort
from app.ports.transcriber import TranscriberPort

logger = get_logger(component="pipeline_engine")

# Sticky, session-lifetime flags — see SessionStore.apply_sticky_flags()'s
# own docstring for the real problem this fixes (a live call's rolling
# transcript window makes contextual_rules/semantic_risk FORGET a signal
# once it ages out of that window, e.g. an authority claim made 20s ago
# plus a financial request just now wouldn't read as connected). Booleans
# OR together (once true, stays true for the rest of the session); the
# one float (urgency) tracks a running max. `urgency_keywords` is handled
# separately (list union) since it isn't a simple scalar.
_STICKY_BOOL_FIELDS = (
    "is_financial_request",
    "authority_claim",
    "semantic_financial_solicitation",
    "semantic_authority_claim",
    "semantic_isolation_request",
)
_STICKY_MAX_FIELDS = ("semantic_urgency_level",)


class SessionStore:
    """In-memory per-session smoothing state.

    Deliberately not persisted anywhere yet — a restart resets every
    session's EMA to a fresh start. That's fine for a single-process
    demo; swap for Redis (keyed by session_id) the moment more than one
    API replica needs to share session state. See docs/architecture.md,
    "What's not built yet".
    """

    def __init__(self) -> None:
        self._last_smoothed: dict[str, float] = {}
        self._sticky_flags: dict[str, dict[str, Any]] = {}

    def get_previous(self, session_id: str) -> Optional[float]:
        return self._last_smoothed.get(session_id)

    def update(self, session_id: str, smoothed_score: float) -> None:
        self._last_smoothed[session_id] = smoothed_score

    def reset(self, session_id: str) -> None:
        self._last_smoothed.pop(session_id, None)
        self._sticky_flags.pop(session_id, None)

    def apply_sticky_flags(self, session_id: str, context: dict[str, Any]) -> dict[str, Any]:
        """Real bug this fixes (found 2026-09-10, live-mic-parity review):
        on the live WebSocket path, `context["transcript"]` is a ROLLING
        ~8s window (app/pipeline/live_transcription.py) — necessary for
        keeping intent/semantic_risk current, but it means a signal
        detected 20s ago (an authority claim, say) silently disappears
        once that phrase ages out of the window, even though a financial
        request just NOW should read as connected to it — exactly the
        classic fraud script contextual_rules.py's own
        combined_authority_financial_pressure rule is built to catch, and
        exactly what it would miss on a long live call without this.

        HOW: rather than growing the transcript itself (a much heavier,
        harder-to-dedupe fix — see #2 Option A in the live-mic-parity plan
        for that alternative), this makes the DERIVED signals sticky for
        the rest of the session: once true, a boolean flag stays true;
        `semantic_urgency_level` tracks a running max; `urgency_keywords`
        is a set union. This re-derives is_financial_request/
        authority_claim/urgency_keywords from whatever `context["transcript"]`
        says THIS window (via the exact same detect_urgency_signals()
        contextual_rules.py itself calls) and feeds the accumulated,
        monotonically-growing result back into those same "manual-
        equivalent" context fields — contextual_rules.py already treats
        `bool(manual) or transcript_derived` as its own precedence rule,
        so persistence falls out of that for free, with ZERO changes to
        that detector. semantic_* fields are stickied directly, since
        semantic_risk_detector.py reads them straight off context with no
        detector-internal re-derivation to route around.

        NO-OP for file-upload: that path's `context["transcript"]` already
        covers the WHOLE call from window 1 onward (Engine.score_call()
        transcribes once, upfront), so every field here is already at its
        final value before scoring starts — sticky-merging it with itself
        changes nothing. This function only has a real effect on the live
        path, where `context` actually changes window to window.
        """
        sticky = self._sticky_flags.setdefault(session_id, {})
        merged = dict(context)

        transcript = context.get("transcript") or ""
        transcript_urgency_keywords, transcript_is_financial, transcript_authority_keywords = (
            detect_urgency_signals(transcript)
        )
        observed = {
            "is_financial_request": bool(context.get("is_financial_request")) or transcript_is_financial,
            "authority_claim": bool(context.get("authority_claim")) or bool(transcript_authority_keywords),
        }
        for field in _STICKY_BOOL_FIELDS:
            if field in observed:
                new_value = observed[field]
            else:
                new_value = bool(context.get(field, False))
            sticky[field] = sticky.get(field, False) or new_value
            merged[field] = sticky[field]

        for field in _STICKY_MAX_FIELDS:
            current = context.get(field)
            if current is not None:
                sticky[field] = max(sticky.get(field, 0.0), float(current))
            if field in sticky:
                merged[field] = sticky[field]

        existing_keywords = sticky.get("urgency_keywords", [])
        for phrase in list(context.get("urgency_keywords") or []) + transcript_urgency_keywords:
            if phrase not in existing_keywords:
                existing_keywords.append(phrase)
        if existing_keywords:
            sticky["urgency_keywords"] = existing_keywords
            merged["urgency_keywords"] = existing_keywords

        return merged


class Engine:
    def __init__(
        self,
        pipeline: Pipeline,
        session_store: Optional[SessionStore] = None,
        history: Optional[HistoryStorePort] = None,
        transcriber: Optional[TranscriberPort] = None,
        live_transcriber: Optional[TranscriberPort] = None,
        intent_classifier: Optional[IntentClassifierPort] = None,
        semantic_risk_classifier: Optional[SemanticRiskClassifierPort] = None,
        live_speaker_embedder: Optional[Any] = None,
    ) -> None:
        self.pipeline = pipeline
        self.sessions = session_store or SessionStore()
        self.history = history
        self.transcriber = transcriber
        # Optional[TranscriberPort] — the LIVE WebSocket path's own
        # transcriber (see app/adapters/registry.py's
        # _build_live_transcriber and config/risk_formula.yaml's
        # `live_transcription:` section), separate from `transcriber`
        # above (file-upload's one-shot transcription). ws_router.py
        # reads this directly off the Engine instance, falling back to
        # `transcriber` when it's None — see that module's own comment
        # at that fallback for why (an unconfigured/unreachable live
        # transcriber degrades rather than disables live transcription).
        self.live_transcriber = live_transcriber
        self.intent_classifier = intent_classifier
        self.semantic_risk_classifier = semantic_risk_classifier
        # Optional[EcapaEmbeddingExtractor] — read by ws_router.py to build
        # a per-session LiveSpeakerTracker (app/pipeline/live_diarization.py).
        # Untyped (Any) here for the same reason Pipeline.live_speaker_embedder
        # is: avoid an eager speechbrain import in this module.
        self.live_speaker_embedder = live_speaker_embedder

    def score_window(self, window: AudioWindow, context: dict[str, Any]) -> FusedScore:
        # See SessionStore.apply_sticky_flags()'s own docstring for why —
        # a no-op for file-upload (context is already whole-call from
        # window 1), a real fix for the live path's rolling-transcript
        # forgetting problem.
        context = self.sessions.apply_sticky_flags(window.session_id, context)
        results = [detector.score(window, context) for detector in self.pipeline.detectors]

        previous = self.sessions.get_previous(window.session_id)
        fused = self.pipeline.fusion.fuse(
            session_id=window.session_id,
            seq=window.seq,
            window_start_ms=window.window_start_ms,
            results=results,
            previous_smoothed_score=previous,
            config=self.pipeline.config,
        )
        self.sessions.update(window.session_id, fused.smoothed_score_0_100)

        logger.info(
            "risk_score_computed",
            session_id=fused.session_id,
            seq=fused.seq,
            window_start_ms=fused.window_start_ms,
            raw_score=fused.raw_score_0_100,
            smoothed_score=fused.smoothed_score_0_100,
            band=fused.band.value,
            formula_version=fused.formula_version,
            third_signal_mode=fused.third_signal_mode,
            components=[
                {
                    "name": c.name,
                    "raw_score": c.raw_score,
                    "weight_configured": c.weight_configured,
                    "weight_effective": c.weight_effective,
                    "contribution": c.contribution,
                    "abstained": c.abstained,
                    "detail": c.detail,
                }
                for c in fused.components
            ],
        )

        if self.history is not None:
            try:
                self.history.save(fused)
            except Exception:  # noqa: BLE001 — history is best-effort, never blocks live scoring
                logger.exception("history_save_failed", session_id=fused.session_id, seq=fused.seq)

        return fused

    def score_windows(
        self, windows: Iterable[AudioWindow], context: dict[str, Any]
    ) -> list[FusedScore]:
        return [self.score_window(w, context) for w in windows]

    def score_call(
        self,
        session_id: str,
        samples,
        sample_rate: int,
        context: dict[str, Any],
    ) -> list[FusedScore]:
        """Convenience: window a whole buffer and score every window.

        Used by the one-shot REST endpoint (POST /v1/score/file) and by
        tests — the natural entrypoint when you have a complete recording
        rather than a live stream.

        If a transcriber is configured, the WHOLE buffer is transcribed
        once here (not per-window — a 2s window is too short for reliable
        ASR, and transcribing it 4x/second of overlap would be wasteful)
        and merged into `context` as `transcript`/`transcript_language`
        before windowing, so every window's third-signal detector sees the
        same transcript. A caller-supplied `transcript` in context is never
        overwritten. Live streaming (score_window/score_windows) has no
        transcription — see app/ports/transcriber.py's scope note.

        If an intent_classifier is ALSO configured, the same transcript is
        classified once here too (same reasoning: a zero-shot NLI forward
        pass is expensive, and IntentRiskDetector only ever reads the
        precomputed `intent_label`/`intent_top_score`/`intent_label_scores`
        this merges into `context` — see that detector's own docstring).
        This detector IS in config/risk_formula.yaml's active `detectors:`
        list (weighted low, 0.15) despite a known calibration problem; see
        app/adapters/intent/zero_shot_intent_classifier.py's HONESTY NOTE
        for what mitigates that risk and what doesn't.

        If a semantic_risk_classifier is ALSO configured, the same
        transcript is analysed once here too (same "expensive, run once"
        reasoning — a local LLM forward pass takes ~1-1.5s, see
        local_llm_semantic_classifier.py). SemanticRiskDetector only ever
        reads the precomputed `semantic_urgency_level`/
        `semantic_financial_solicitation`/`semantic_authority_claim`/
        `semantic_isolation_request`/`semantic_reasoning` this merges
        into `context` — see that detector's own docstring. See
        config/risk_formula.yaml's `semantic_risk` entry for whether and
        at what weight this is currently active.
        """
        self.sessions.reset(session_id)

        if self.transcriber is not None and not context.get("transcript"):
            transcript_result = self.transcriber.transcribe(samples, sample_rate)
            logger.info(
                "transcript_computed",
                session_id=session_id,
                detector_name=transcript_result.detector_name,
                language=transcript_result.language,
                transcript_length=len(transcript_result.text),
            )
            if transcript_result.text:
                context = {
                    **context,
                    "transcript": transcript_result.text,
                    "transcript_language": transcript_result.language,
                }

        if (
            self.intent_classifier is not None
            and context.get("transcript")
            and "intent_label" not in context
        ):
            intent_result = self.intent_classifier.classify(context["transcript"])
            logger.info(
                "intent_classified",
                session_id=session_id,
                detector_name=intent_result.detector_name,
                top_label=intent_result.top_label,
                top_score=intent_result.top_score,
            )
            context = {
                **context,
                "intent_label": intent_result.top_label,
                "intent_top_score": intent_result.top_score,
                "intent_label_scores": intent_result.label_scores,
            }

        if (
            self.semantic_risk_classifier is not None
            and context.get("transcript")
            and "semantic_urgency_level" not in context
        ):
            semantic_result = self.semantic_risk_classifier.analyze(context["transcript"])
            logger.info(
                "semantic_risk_analyzed",
                session_id=session_id,
                detector_name=semantic_result.detector_name,
                urgency_level=semantic_result.urgency_level,
                financial_solicitation=semantic_result.financial_solicitation,
                authority_claim=semantic_result.authority_claim,
                isolation_request=semantic_result.isolation_request,
            )
            context = {
                **context,
                "semantic_urgency_level": semantic_result.urgency_level,
                "semantic_financial_solicitation": semantic_result.financial_solicitation,
                "semantic_authority_claim": semantic_result.authority_claim,
                "semantic_isolation_request": semantic_result.isolation_request,
                "semantic_reasoning": semantic_result.reasoning,
            }

        windowing_cfg = self.pipeline.config["windowing"]
        windows = make_windows(
            session_id=session_id,
            samples=samples,
            sample_rate=sample_rate,
            window_ms=windowing_cfg["window_ms"],
            hop_ms=windowing_cfg["hop_ms"],
        )
        return self.score_windows(windows, context)

    def diarize_and_score(
        self,
        diarizer: DiarizerPort,
        base_session_id: str,
        samples: np.ndarray,
        sample_rate: int,
        context: dict[str, Any],
    ) -> list[SpeakerCallResult]:
        """Splits `samples` by speaker and reruns each speaker's audio
        through score_call() — a derived session_id
        ("{base}::{speaker_label}") means every window still flows through
        the normal fusion + history-save machinery with no special-casing
        there, and is retrievable afterward exactly like any other session
        (GET /v1/sessions/{base}::{speaker_label}).

        Added 2026-09-10, moved here from what used to be http_router.py's
        own _diarize_and_score() so BOTH the file-upload path
        (POST /v1/score/file?diarize=true) and the live WebSocket path's
        "diarize on hangup" (app/api/ws_router.py's own docstring explains
        why a live call's per-speaker breakdown only exists once the call
        ends, not window-by-window — that's app/pipeline/live_diarization.py's
        job instead, a genuinely different, narrower, real-time feature)
        call the exact same code, instead of it being duplicated.

        Also persists each speaker's segment_count/total_duration_ms via
        HistoryStorePort.save_speaker_summary() (best-effort, same
        never-break-the-caller reasoning as score_window()'s own
        history.save() error handling) — the one part of the result that
        isn't re-derivable from FusedScore history alone, and what powers
        GET /v1/sessions/{id}/speakers for discovering which derived
        sessions exist for a given call.
        """
        segments = diarizer.diarize(samples, sample_rate)
        if not segments:
            return []

        speaker_order: list[str] = []
        speaker_chunks: dict[str, list[np.ndarray]] = {}
        speaker_duration_ms: dict[str, int] = {}
        for seg in segments:
            if seg.speaker_label not in speaker_chunks:
                speaker_chunks[seg.speaker_label] = []
                speaker_duration_ms[seg.speaker_label] = 0
                speaker_order.append(seg.speaker_label)
            start_sample = int(seg.start_ms / 1000 * sample_rate)
            end_sample = int(seg.end_ms / 1000 * sample_rate)
            speaker_chunks[seg.speaker_label].append(samples[start_sample:end_sample])
            speaker_duration_ms[seg.speaker_label] += seg.end_ms - seg.start_ms

        results: list[SpeakerCallResult] = []
        for speaker_label in speaker_order:
            speaker_samples = np.concatenate(speaker_chunks[speaker_label])
            speaker_session_id = f"{base_session_id}::{speaker_label}"
            speaker_fused = self.score_call(
                session_id=speaker_session_id,
                samples=speaker_samples,
                sample_rate=sample_rate,
                context=context,
            )
            if not speaker_fused:
                continue  # this speaker's total voiced audio was too short to window
            segment_count = len(speaker_chunks[speaker_label])
            total_duration_ms = speaker_duration_ms[speaker_label]
            results.append(
                SpeakerCallResult(
                    speaker_label=speaker_label,
                    session_id=speaker_session_id,
                    segment_count=segment_count,
                    total_duration_ms=total_duration_ms,
                    fused_scores=speaker_fused,
                )
            )
            if self.history is not None:
                try:
                    self.history.save_speaker_summary(
                        SpeakerCallSummary(
                            base_session_id=base_session_id,
                            speaker_label=speaker_label,
                            speaker_session_id=speaker_session_id,
                            segment_count=segment_count,
                            total_duration_ms=total_duration_ms,
                        )
                    )
                except Exception:  # noqa: BLE001 — best-effort, never breaks the caller
                    logger.exception("speaker_summary_save_failed", base_session_id=base_session_id)
        return results
