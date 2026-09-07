import numpy as np

from app.adapters.enrollment.sqlite_enrollment_store import SqliteEnrollmentStore

_MODEL = "speechbrain/spkrec-ecapa-voxceleb"


def _embedding(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 1, 192).astype(np.float32)


def test_enroll_and_get_embedding_round_trips(tmp_path):
    store = SqliteEnrollmentStore(str(tmp_path / "voiceprints.db"))
    embedding = _embedding(0)

    store.enroll("alice", embedding, _MODEL)
    round_tripped = store.get_embedding("alice")

    np.testing.assert_allclose(round_tripped, embedding, rtol=1e-6)


def test_get_unknown_identity_returns_none(tmp_path):
    store = SqliteEnrollmentStore(str(tmp_path / "voiceprints.db"))
    assert store.get_embedding("nobody") is None


def test_re_enroll_overwrites_embedding(tmp_path):
    store = SqliteEnrollmentStore(str(tmp_path / "voiceprints.db"))
    store.enroll("alice", _embedding(0), _MODEL)
    store.enroll("alice", _embedding(1), _MODEL)

    round_tripped = store.get_embedding("alice")

    np.testing.assert_allclose(round_tripped, _embedding(1), rtol=1e-6)
    assert len(store.list_enrollments()) == 1  # still only one row for alice


def test_list_enrollments_summarises_without_leaking_embeddings(tmp_path):
    store = SqliteEnrollmentStore(str(tmp_path / "voiceprints.db"))
    store.enroll("alice", _embedding(0), _MODEL)
    store.enroll("bob", _embedding(1), _MODEL)

    summaries = store.list_enrollments()

    identities = {s.identity for s in summaries}
    assert identities == {"alice", "bob"}
    for s in summaries:
        assert s.embedding_model == _MODEL
        assert s.enrolled_at
        assert s.updated_at
        assert not hasattr(s, "embedding")  # never exposes the raw vector


def test_delete_removes_and_reports_existence(tmp_path):
    store = SqliteEnrollmentStore(str(tmp_path / "voiceprints.db"))
    store.enroll("alice", _embedding(0), _MODEL)

    assert store.delete("alice") is True
    assert store.get_embedding("alice") is None
    assert store.delete("alice") is False  # already gone


def test_store_persists_across_reopening_the_same_file(tmp_path):
    db_path = str(tmp_path / "voiceprints.db")
    SqliteEnrollmentStore(db_path).enroll("alice", _embedding(0), _MODEL)

    reopened = SqliteEnrollmentStore(db_path)

    np.testing.assert_allclose(reopened.get_embedding("alice"), _embedding(0), rtol=1e-6)
