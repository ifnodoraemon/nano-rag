"""source_key must separate unrelated documents without splitting version families."""
from __future__ import annotations

from app.ingestion.metadata import build_source_key


def test_same_leaf_directory_in_different_subtrees_does_not_collide() -> None:
    """Regression: only the leaf parent segment was used as identity.

    ``teamA/reports/spec.pdf`` and ``teamB/reports/spec.pdf`` both became
    "reports spec", so the version ledger treated one as a superseded version of
    the other and the version filter dropped it from retrieval - an ingested,
    indexed document that could never be found.
    """
    a = build_source_key("corpus/teamA/reports/spec.pdf", "spec")
    b = build_source_key("corpus/teamB/reports/spec.pdf", "spec")
    assert a != b, f"distinct documents collided on source_key {a!r}"


def test_version_family_in_one_directory_still_groups() -> None:
    """The whole point of source_key is that versions of one document group."""
    v1 = build_source_key("policies/travel_policy_v1.pdf", "travel policy")
    v2 = build_source_key("policies/travel_policy_v2.pdf", "travel policy")
    assert v1 == v2


def test_same_directory_different_titles_stay_separate() -> None:
    assert build_source_key("policies/travel.pdf", "travel") != build_source_key(
        "policies/leave.pdf", "leave"
    )


def test_root_level_file_has_no_parent_segment() -> None:
    assert build_source_key("handbook.md", "handbook") == "handbook"


def test_key_is_normalised() -> None:
    assert build_source_key("Data/Raw/Handbook.md", "Employee Handbook") == (
        "data raw employee handbook"
    )
