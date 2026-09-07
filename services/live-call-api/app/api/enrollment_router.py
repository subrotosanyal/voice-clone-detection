"""Voiceprint enrollment endpoints.

Reaches the SAME VoiceprintConsistencyDetector instance the scoring
pipeline already built (app.state.voiceprint_detector, wired up in
app/main.py's lifespan) rather than constructing a second embedding model —
see app/adapters/detectors/voiceprint_consistency.py's docstring on why
that instance is shared between scoring and enrollment.

    curl -F "identity=alice" -F "file=@enroll_sample.wav" \\
         http://localhost:8000/v1/enroll

Only ever stores the derived embedding vector, never the raw audio — see
app/adapters/enrollment/sqlite_enrollment_store.py's schema note.
"""
from __future__ import annotations

import io

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.api.schemas import EnrollmentSummaryOut, EnrollResponse

router = APIRouter()


def _get_voiceprint_detector(request: Request):
    detector = getattr(request.app.state, "voiceprint_detector", None)
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "voiceprint consistency detector is not configured on this instance — "
                "third_signal.consistency_class in config/risk_formula.yaml must be "
                "app.adapters.detectors.voiceprint_consistency:VoiceprintConsistencyDetector"
            ),
        )
    return detector


@router.post("/v1/enroll", response_model=EnrollResponse)
async def enroll(
    request: Request,
    identity: str = Form(...),
    file: UploadFile = File(...),
) -> EnrollResponse:
    detector = _get_voiceprint_detector(request)

    raw_bytes = await file.read()
    try:
        samples, sample_rate = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
    except Exception as exc:  # noqa: BLE001 — surfaced to the caller verbatim
        raise HTTPException(status_code=400, detail=f"could not decode audio file: {exc}") from exc

    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)
    if samples.size == 0:
        raise HTTPException(status_code=422, detail="audio file is empty")

    detector.enroll(identity, samples, int(sample_rate))
    return EnrollResponse(
        identity=identity,
        embedding_model=detector.embedding_model_source,
        message=f"voiceprint enrolled for {identity!r}",
    )


@router.get("/v1/enrollments", response_model=list[EnrollmentSummaryOut])
def list_enrollments(request: Request) -> list[EnrollmentSummaryOut]:
    detector = _get_voiceprint_detector(request)
    return [EnrollmentSummaryOut.from_domain(e) for e in detector.list_enrollments()]


@router.delete("/v1/enrollments/{identity}")
def delete_enrollment(identity: str, request: Request) -> dict:
    detector = _get_voiceprint_detector(request)
    deleted = detector.delete_enrollment(identity)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"no enrollment found for identity {identity!r}")
    return {"deleted": True, "identity": identity}
