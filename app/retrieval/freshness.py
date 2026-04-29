from __future__ import annotations

import re
from dataclasses import dataclass

from app.retrieval.filters import parse_date
from app.vectorstore.repository import SearchHit

@dataclass(frozen=True)
class FreshnessPolicy:
    enabled: bool = True


def prioritize_fresh_hits(
    hits: list[SearchHit],
    policy: FreshnessPolicy | None = None,
) -> list[SearchHit]:
    policy = policy or FreshnessPolicy()
    if not policy.enabled or not hits:
        return hits

    groups: dict[str, list[tuple[int, SearchHit]]] = {}
    passthrough: list[tuple[int, SearchHit]] = []
    for index, hit in enumerate(hits):
        group_key = _group_key(hit)
        if group_key is None:
            passthrough.append((index, _annotate_hit(hit, None, None, None)))
            continue
        groups.setdefault(group_key, []).append((index, hit))

    primary: list[tuple[int, SearchHit]] = list(passthrough)
    stale: list[tuple[int, SearchHit]] = []
    for group_hits in groups.values():
        ranked = sorted(
            group_hits,
            key=lambda item: _freshness_sort_key(item[1]),
            reverse=True,
        )
        winner_index, winner_hit = ranked[0]
        primary.append((winner_index, _annotate_hit(winner_hit, "primary", True, 0)))
        for stale_rank, (stale_index, stale_hit) in enumerate(ranked[1:], start=1):
            stale.append(
                (
                    stale_index,
                    _annotate_hit(
                        stale_hit,
                        "supplemental",
                        False,
                        stale_rank,
                    ),
                )
            )

    primary.sort(key=lambda item: item[0])
    return [hit for _, hit in primary]


def _group_key(hit: SearchHit) -> str | None:
    metadata = hit.chunk.metadata or {}
    if metadata.get("wiki_kind") == "topic":
        return None
    source_key_value = metadata.get("source_key")
    if source_key_value is None:
        return None
    source_key = str(source_key_value).strip()
    if not source_key:
        return None
    section = metadata.get("section_path_text")
    if not isinstance(section, str) or not section.strip():
        section_path = metadata.get("section_path")
        if isinstance(section_path, list):
            section = " > ".join(str(item) for item in section_path if str(item).strip())
        else:
            section = hit.chunk.title or ""
    return f"{metadata.get('wiki_kind') or 'raw'}|{source_key}|{section.strip().lower()}"

def _freshness_sort_key(hit: SearchHit) -> tuple[int, object, tuple[int, ...], float]:
    metadata = hit.chunk.metadata or {}
    effective_date = parse_date(metadata.get("effective_date"))
    version = _version_key(metadata.get("version"))
    has_date = 1 if effective_date else 0
    return (
        has_date,
        effective_date or "",
        version,
        hit.score,
    )


def _version_key(value: object) -> tuple[int, ...]:
    if not isinstance(value, str):
        return ()
    parts = re.findall(r"\d+", value)
    if not parts:
        return ()
    return tuple(int(part) for part in parts)


def _annotate_hit(
    hit: SearchHit,
    freshness_tier: str | None,
    is_latest_version: bool | None,
    freshness_rank: int | None,
) -> SearchHit:
    metadata = {**(hit.chunk.metadata or {})}
    if freshness_tier is not None:
        metadata["freshness_tier"] = freshness_tier
    if is_latest_version is not None:
        metadata["is_latest_version"] = is_latest_version
    if freshness_rank is not None:
        metadata["freshness_rank"] = freshness_rank
    return SearchHit(
        chunk=hit.chunk.model_copy(update={"metadata": metadata}),
        score=hit.score,
    )
