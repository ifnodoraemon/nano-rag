import json

from app.core.tracing import FeedbackStore


def test_feedback_store_loads_persisted_records(tmp_path) -> None:
    feedback_dir = tmp_path / "feedback"
    feedback_dir.mkdir()
    (feedback_dir / "fb-1.json").write_text(
        json.dumps(
            {
                "feedback_id": "fb-1",
                "trace_id": "trace-1",
                "rating": "up",
                "kb_id": "default",
                "created_at": 1.0,
            }
        ),
        encoding="utf-8",
    )

    store = FeedbackStore(persist_dir=feedback_dir)

    assert store.list()[0].feedback_id == "fb-1"


def test_feedback_store_prunes_old_persisted_files(tmp_path) -> None:
    feedback_dir = tmp_path / "feedback"
    store = FeedbackStore(max_records=2, persist_dir=feedback_dir)

    store.save({"feedback_id": "fb-1", "trace_id": "trace-1", "rating": "up", "kb_id": "default", "created_at": 1.0})
    store.save({"feedback_id": "fb-2", "trace_id": "trace-2", "rating": "up", "kb_id": "default", "created_at": 2.0})
    store.save({"feedback_id": "fb-3", "trace_id": "trace-3", "rating": "up", "kb_id": "default", "created_at": 3.0})

    persisted = sorted(path.name for path in feedback_dir.glob("*.json"))
    assert persisted == ["fb-2.json", "fb-3.json"]
