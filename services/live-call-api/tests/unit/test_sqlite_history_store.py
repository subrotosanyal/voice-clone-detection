from app.adapters.history.sqlite_store import SqliteHistoryStore
from app.domain.models import Band, ComponentContribution, FusedScore, SpeakerCallSummary


def _fused(session_id: str, seq: int, score: float, band: Band = Band.LOW) -> FusedScore:
    return FusedScore(
        session_id=session_id,
        seq=seq,
        window_start_ms=seq * 500,
        raw_score_0_100=score,
        smoothed_score_0_100=score,
        band=band,
        formula_version="test-1",
        third_signal_mode="contextual",
        recommended_action="None — monitor silently.",
        components=[
            ComponentContribution(
                name="acoustic",
                raw_score=0.1,
                weight_configured=0.6,
                weight_effective=0.6,
                contribution=0.06,
                abstained=False,
                detail={"spectral_flatness": 0.1},
            ),
            ComponentContribution(
                name="prosodic",
                raw_score=None,
                weight_configured=0.2,
                weight_effective=0.0,
                contribution=0.0,
                abstained=True,
                detail={},
            ),
        ],
    )


def test_save_and_get_session_round_trips_everything(tmp_path):
    store = SqliteHistoryStore(str(tmp_path / "sessions.db"))
    store.save(_fused("s1", 0, 10.0))
    store.save(_fused("s1", 1, 12.0))

    trace = store.get_session("s1")

    assert len(trace) == 2
    assert [f.seq for f in trace] == [0, 1]
    assert trace[0].raw_score_0_100 == 10.0
    # abstained component round-trips correctly, including a None raw_score
    prosodic = next(c for c in trace[0].components if c.name == "prosodic")
    assert prosodic.abstained is True
    assert prosodic.raw_score is None
    acoustic = next(c for c in trace[0].components if c.name == "acoustic")
    assert acoustic.detail == {"spectral_flatness": 0.1}


def test_get_unknown_session_returns_empty_list(tmp_path):
    store = SqliteHistoryStore(str(tmp_path / "sessions.db"))
    assert store.get_session("does-not-exist") == []


def test_list_sessions_orders_most_recent_first_and_summarises(tmp_path):
    store = SqliteHistoryStore(str(tmp_path / "sessions.db"))
    store.save(_fused("older", 0, 10.0, Band.LOW))
    store.save(_fused("newer", 0, 20.0, Band.LOW))
    store.save(_fused("newer", 1, 80.0, Band.HIGH))  # final window of "newer"

    sessions = store.list_sessions()

    ids = [s.session_id for s in sessions]
    assert set(ids) == {"older", "newer"}
    newer = next(s for s in sessions if s.session_id == "newer")
    assert newer.window_count == 2
    assert newer.final_smoothed_score == 80.0  # last seq, not first
    assert newer.final_band == Band.HIGH


def test_list_sessions_respects_limit(tmp_path):
    store = SqliteHistoryStore(str(tmp_path / "sessions.db"))
    for i in range(5):
        store.save(_fused(f"s{i}", 0, 10.0))

    assert len(store.list_sessions(limit=2)) == 2


def test_delete_session_removes_it_and_reports_existence(tmp_path):
    store = SqliteHistoryStore(str(tmp_path / "sessions.db"))
    store.save(_fused("s1", 0, 10.0))

    assert store.delete_session("s1") is True
    assert store.get_session("s1") == []
    assert store.delete_session("s1") is False  # already gone


def test_store_persists_across_reopening_the_same_file(tmp_path):
    db_path = str(tmp_path / "sessions.db")
    SqliteHistoryStore(db_path).save(_fused("s1", 0, 42.0))

    reopened = SqliteHistoryStore(db_path)
    trace = reopened.get_session("s1")

    assert len(trace) == 1
    assert trace[0].raw_score_0_100 == 42.0


def test_save_and_list_speaker_summaries_round_trip(tmp_path):
    store = SqliteHistoryStore(str(tmp_path / "sessions.db"))
    store.save_speaker_summary(
        SpeakerCallSummary(
            base_session_id="live-1",
            speaker_label="speaker_1",
            speaker_session_id="live-1::speaker_1",
            segment_count=3,
            total_duration_ms=4500,
        )
    )
    store.save_speaker_summary(
        SpeakerCallSummary(
            base_session_id="live-1",
            speaker_label="speaker_2",
            speaker_session_id="live-1::speaker_2",
            segment_count=2,
            total_duration_ms=3000,
        )
    )

    summaries = store.list_speaker_summaries("live-1")

    assert [s.speaker_label for s in summaries] == ["speaker_1", "speaker_2"]  # first-seen order
    assert summaries[0].segment_count == 3
    assert summaries[0].total_duration_ms == 4500
    assert summaries[0].speaker_session_id == "live-1::speaker_1"


def test_list_speaker_summaries_empty_for_a_session_never_diarized(tmp_path):
    store = SqliteHistoryStore(str(tmp_path / "sessions.db"))
    store.save(_fused("s1", 0, 10.0))  # a normal session, never diarized

    assert store.list_speaker_summaries("s1") == []


def test_speaker_summaries_are_scoped_to_their_base_session(tmp_path):
    store = SqliteHistoryStore(str(tmp_path / "sessions.db"))
    store.save_speaker_summary(
        SpeakerCallSummary(
            base_session_id="live-1",
            speaker_label="speaker_1",
            speaker_session_id="live-1::speaker_1",
            segment_count=1,
            total_duration_ms=1000,
        )
    )

    assert store.list_speaker_summaries("live-2") == []
