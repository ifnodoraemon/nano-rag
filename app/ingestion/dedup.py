"""Content-addressed document identity, and duplicate consolidation.

Why this exists
---------------
Document identity used to be path-only (``doc_id = sha256(kb_id, source_path)``),
which is correct for re-ingesting the same path but says nothing about the same
*file* arriving under a new path: an upload, a reorganized corpus, or a renamed
file produced a second document with byte-identical content. Every duplicate
occupies a discovery slot and splits BM25 scores, so the copies crowd the real
answer out of the top-k — a measurable recall loss, not just clutter.

Two halves:

- :func:`find_duplicate_documents` — group documents in a KB by a content
  fingerprint and report the redundant ones. Used by the ingest path (to
  supersede a moved file) and by ``scripts/dedup_corpus.py`` (to clean up a
  corpus that already accumulated duplicates).
- :func:`remove_document_everywhere` — delete one document from all three
  stores (parsed artifact + attachments, graph rows, wiki page) so a removal
  cannot leave a half-deleted document that retrieval still half-sees.

The fingerprint is ``sha256(document.content)``, not the stored
``source_content_hash``: legacy artifacts predate that field (they store
``None``), so keying on it would silently find no duplicates precisely in the
corpora that need cleaning. Content is present in every artifact.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.exceptions import StoreError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.retrieval.graph_store import GraphStore
    from app.wiki.compiler import WikiCompiler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentFingerprint:
    """One document as dedup sees it."""

    doc_id: str
    kb_id: str
    content_fingerprint: str
    source_path: str
    source_content_hash: str | None
    artifact_mtime: float

    @property
    def is_current_format(self) -> bool:
        """Artifacts written by the current pipeline carry a content hash."""
        return bool(self.source_content_hash)

    @property
    def is_stable_doc_id(self) -> bool:
        return self.doc_id.startswith("doc-")


def content_fingerprint(text: str) -> str:
    """Stable fingerprint of a document's text."""
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def read_document_fingerprint(artifact: Path) -> DocumentFingerprint | None:
    """Read one parsed artifact, or ``None`` when it carries no document.

    A corrupt artifact raises: it is a data-integrity failure and skipping it
    would hide the very file an operator needs to look at.
    """
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StoreError(
            f"parsed artifact {artifact.name} is unreadable: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise StoreError(f"parsed artifact {artifact.name} is not a JSON object")
    document = payload.get("document")
    if not isinstance(document, dict):
        return None
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    doc_id = str(document.get("doc_id") or artifact.stem)
    content = str(document.get("content") or "")
    if not content:
        # An artifact with no text (media-only) cannot be fingerprinted by
        # content; it is not a duplicate candidate.
        return None
    try:
        mtime = artifact.stat().st_mtime
    except OSError:
        mtime = 0.0
    return DocumentFingerprint(
        doc_id=doc_id,
        kb_id=str(metadata.get("kb_id") or "default"),
        content_fingerprint=content_fingerprint(content),
        source_path=str(document.get("source_path") or ""),
        source_content_hash=(
            str(metadata["source_content_hash"])
            if metadata.get("source_content_hash")
            else None
        ),
        artifact_mtime=mtime,
    )


def iter_fingerprints(parsed_dir: Path) -> list[DocumentFingerprint]:
    """Fingerprint every artifact in a parsed directory."""
    if not parsed_dir.exists():
        return []
    fingerprints: list[DocumentFingerprint] = []
    for artifact in sorted(parsed_dir.glob("*.json")):
        fingerprint = read_document_fingerprint(artifact)
        if fingerprint is not None:
            fingerprints.append(fingerprint)
    return fingerprints


def _path_components(value: str) -> list[str]:
    return [part for part in value.replace("\\", "/").split("/") if part]


def same_source_file(left: str, right: str) -> bool:
    """True when two source paths point at the same file under different roots.

    An artifact written by a bare local script records a repo-relative path
    (``data/raw/x.md``) while the container records an absolute one
    (``/workspace/data/raw/x.md``). Comparing the trailing components treats
    those as the same file without confusing ``reports/spec.pdf`` with
    ``contracts/spec.pdf`` (a single shared basename is not enough).
    """
    left_parts = _path_components(left)
    right_parts = _path_components(right)
    shared = min(len(left_parts), len(right_parts))
    if shared < 2:
        return False
    return left_parts[-shared:] == right_parts[-shared:]


def _keep_priority(fingerprint: DocumentFingerprint) -> tuple:
    """Order documents so the most trustworthy copy sorts first.

    Prefers the current artifact format (it carries provenance), then the
    stable ``doc-`` identity, then the newest artifact. Deterministic, so a
    cleanup keeps the same copy every run.
    """
    return (
        1 if fingerprint.is_current_format else 0,
        1 if fingerprint.is_stable_doc_id else 0,
        fingerprint.artifact_mtime,
        fingerprint.doc_id,
    )


@dataclass(frozen=True)
class DuplicateGroup:
    kb_id: str
    
    # Either an identical-content group ("content") or the same source file
    # recorded under two identities after a doc_id-rule change ("stale_identity").
    reason: str
    identity: str
    keep: DocumentFingerprint
    redundant: tuple[DocumentFingerprint, ...]


def find_duplicate_documents(
    fingerprints: list[DocumentFingerprint],
) -> list[DuplicateGroup]:
    """Group same-content documents within a KB; keep one, mark the rest."""
    grouped: dict[tuple[str, str], list[DocumentFingerprint]] = {}
    for fingerprint in fingerprints:
        grouped.setdefault(
            (fingerprint.kb_id, fingerprint.content_fingerprint), []
        ).append(fingerprint)

    groups: list[DuplicateGroup] = []
    survivors: list[DocumentFingerprint] = [
        fingerprint
        for key, members in grouped.items()
        if len(members) == 1
        for fingerprint in members
    ]
    for (kb_id, content), members in sorted(grouped.items()):
        if len(members) < 2:
            continue
        ordered = sorted(members, key=_keep_priority, reverse=True)
        groups.append(
            DuplicateGroup(
                kb_id=kb_id,
                reason="content",
                identity=content,
                keep=ordered[0],
                redundant=tuple(ordered[1:]),
            )
        )
        # Only the kept copy takes part in the identity pass, so the two rules
        # never report the same document twice.
        survivors.append(ordered[0])
    groups.extend(_stale_identity_groups(survivors))
    return groups


def _stale_identity_groups(
    fingerprints: list[DocumentFingerprint],
) -> list[DuplicateGroup]:
    """Group artifacts that describe the same source file under two identities.

    After a doc_id-rule change (or any re-ingest that produced a new id), the
    old artifact and the new one coexist: same KB, same source file, same
    title, different doc_id. Their text usually differs too (different parser
    version), so content matching cannot see them — and they compete for the
    same discovery slot exactly like a content duplicate.

    Deliberately narrow: a group only forms when one member carries the stable
    ``doc-`` id scheme and the others carry an opaque legacy id. That is the
    upgrade case and nothing else, so a normal corpus cannot be pruned by a
    path heuristic.

    The discriminator is the doc_id scheme, not the presence of
    ``source_content_hash``: artifacts produced by intermediate builds have the
    stable id but predate the hash field, and they are exactly the copies that
    must lose to the current one.
    """
    # Bucket by KB first, then match source files pairwise: the path
    # normalization is a comparison, not a hash key.
    by_kb: dict[str, list[DocumentFingerprint]] = {}
    for fingerprint in fingerprints:
        if fingerprint.source_path:
            by_kb.setdefault(fingerprint.kb_id, []).append(fingerprint)

    groups: list[DuplicateGroup] = []
    for kb_id, members in sorted(by_kb.items()):
        claimed: set[str] = set()
        for member in members:
            if member.doc_id in claimed or not member.is_stable_doc_id:
                continue
            twins = [
                other
                for other in members
                if other.doc_id != member.doc_id
                and other.doc_id not in claimed
                and not other.is_stable_doc_id
                and same_source_file(member.source_path, other.source_path)
            ]
            if not twins:
                continue
            siblings = sorted(twins, key=_keep_priority, reverse=True)
            claimed.update(sibling.doc_id for sibling in siblings)
            groups.append(
                DuplicateGroup(
                    kb_id=kb_id,
                    reason="stale_identity",
                    identity=member.source_path,
                    keep=member,
                    redundant=tuple(siblings),
                )
            )
    return groups


def find_duplicates_for_content(
    parsed_dir: Path,
    *,
    kb_id: str,
    content: str,
    exclude_doc_id: str | None = None,
) -> list[DocumentFingerprint]:
    """Existing documents in ``kb_id`` with the same content as ``content``.

    Used by the ingest path so a file that arrives under a new path supersedes
    the old copy instead of adding a second one.
    """
    if not content:
        return []
    wanted = content_fingerprint(content)
    matches = [
        fingerprint
        for fingerprint in iter_fingerprints(parsed_dir)
        if fingerprint.kb_id == kb_id
        and fingerprint.content_fingerprint == wanted
        and fingerprint.doc_id != exclude_doc_id
    ]
    return sorted(matches, key=_keep_priority, reverse=True)


def remove_document_everywhere(
    *,
    doc_id: str,
    kb_id: str,
    parsed_dir: Path,
    graph_store: "GraphStore | None" = None,
    wiki_compiler: "WikiCompiler | None" = None,
) -> list[str]:
    """Delete one document from every store that holds it.

    Returns the list of stores that were touched. Deletions are independent:
    a missing wiki page or an unreachable graph store must not leave the
    artifact behind, because the artifact is what discovery indexes.
    """
    touched: list[str] = []

    artifact = parsed_dir / f"{doc_id}.json"
    if artifact.exists():
        artifact.unlink()
        touched.append("artifact")

    attachments = parsed_dir / "attachments" / doc_id
    if attachments.exists():
        shutil.rmtree(attachments)
        touched.append("attachments")

    if graph_store is not None:
        graph_store.delete_document(doc_id=doc_id, kb_id=kb_id)
        touched.append("graph")

    if wiki_compiler is not None:
        wiki_compiler.remove_document(doc_id)
        touched.append("wiki")

    return touched
