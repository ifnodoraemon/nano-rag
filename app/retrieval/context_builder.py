import re

from app.vectorstore.repository import SearchHit

DEFAULT_BUCKET_ORDER = ["topic", "raw", "source", "index"]
DEFAULT_EVIDENCE_ORDER = ["primary", "supporting", "conflicting"]
CJK_SEQUENCE_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+")
WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _context_bucket(context: dict[str, object]) -> str:
    wiki_kind = context.get("wiki_kind")
    if isinstance(wiki_kind, str) and wiki_kind in {"topic", "source", "index"}:
        return wiki_kind
    return "raw"


def _evidence_role(context: dict[str, object]) -> str:
    if context.get("wiki_status") == "conflicting":
        return "conflicting"
    if context.get("freshness_tier") == "primary":
        return "primary"
    if context.get("wiki_kind") == "topic" and context.get("wiki_status") == "stable":
        return "primary"
    return "supporting"


def _order_contexts_by_evidence(contexts: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {role: [] for role in DEFAULT_EVIDENCE_ORDER}
    extras: list[dict[str, object]] = []
    for context in contexts:
        role = str(context.get("evidence_role") or "supporting")
        if role in grouped:
            grouped[role].append(context)
        else:
            extras.append(context)
    ordered: list[dict[str, object]] = []
    for role in DEFAULT_EVIDENCE_ORDER:
        ordered.extend(grouped[role])
    ordered.extend(extras)
    return ordered


def _with_citation_labels(contexts: list[dict[str, object]]) -> list[dict[str, object]]:
    labeled: list[dict[str, object]] = []
    for index, context in enumerate(contexts, start=1):
        public_context = {
            key: value for key, value in context.items() if key != "_dedupe_key"
        }
        labeled.append({**public_context, "citation_label": f"C{index}"})
    return labeled


def _query_terms(query: str | None) -> set[str]:
    if not query:
        return set()
    terms = {token.lower() for token in WORD_RE.findall(query) if len(token) >= 2}
    for match in CJK_SEQUENCE_RE.finditer(query):
        chars = match.group()
        for size in range(2, min(5, len(chars) + 1)):
            for index in range(0, len(chars) - size + 1):
                terms.add(chars[index : index + size])
    return terms


def _context_search_text(context: dict[str, object]) -> str:
    parts = [
        context.get("text"),
        context.get("title"),
        context.get("source"),
        context.get("source_key"),
    ]
    section_path = context.get("section_path")
    if isinstance(section_path, list):
        parts.extend(section_path)
    return " ".join(str(part) for part in parts if part)


def _coverage_for_context(
    context: dict[str, object], terms: set[str]
) -> set[str]:
    search_text = _context_search_text(context).lower()
    return {term for term in terms if term.lower() in search_text}


def _dedupe_key(context: dict[str, object]) -> str:
    return str(
        context.get("_dedupe_key")
        or context.get("parent_chunk_id")
        or context.get("chunk_id")
    )


def _least_valuable_selected_index(
    selected: list[dict[str, object]],
    coverages: dict[str, set[str]],
) -> int:
    selected_keys = [_dedupe_key(context) for context in selected]
    scores: list[tuple[int, float, int]] = []
    for index, key in enumerate(selected_keys):
        other_coverage: set[str] = set()
        for other_key in selected_keys:
            if other_key == key:
                continue
            other_coverage.update(coverages.get(other_key, set()))
        unique_terms = coverages.get(key, set()) - other_coverage
        raw_score = float(selected[index].get("score", 0.0) or 0.0)
        scores.append((len(unique_terms), raw_score, -index))
    return min(range(len(scores)), key=lambda index: scores[index])


def _promote_query_coverage(
    selected: list[dict[str, object]],
    contexts: list[dict[str, object]],
    query: str | None,
) -> list[dict[str, object]]:
    terms = _query_terms(query)
    if not terms or not selected:
        return selected

    coverages = {
        _dedupe_key(context): _coverage_for_context(context, terms)
        for context in contexts
    }
    coverable_terms: set[str] = set()
    for coverage in coverages.values():
        coverable_terms.update(coverage)
    if not coverable_terms:
        return selected

    selected_keys = {_dedupe_key(context) for context in selected}
    selected_coverage: set[str] = set()
    for key in selected_keys:
        selected_coverage.update(coverages.get(key, set()))
    missing_terms = coverable_terms - selected_coverage
    if not missing_terms:
        return selected

    promoted = list(selected)
    for candidate in contexts:
        candidate_key = _dedupe_key(candidate)
        if candidate_key in selected_keys:
            continue
        candidate_missing = coverages.get(candidate_key, set()) & missing_terms
        if not candidate_missing:
            continue
        replace_index = _least_valuable_selected_index(promoted, coverages)
        replaced_key = _dedupe_key(promoted[replace_index])
        promoted[replace_index] = candidate
        selected_keys.remove(replaced_key)
        selected_keys.add(candidate_key)
        selected_coverage = set()
        for key in selected_keys:
            selected_coverage.update(coverages.get(key, set()))
        missing_terms = coverable_terms - selected_coverage
        if not missing_terms:
            break
    return promoted


def _is_truncated_parent_text(value: object) -> bool:
    return isinstance(value, str) and value.rstrip().endswith("...")


def _context_text_and_dedupe_key(hit: SearchHit) -> tuple[str, str]:
    metadata = hit.chunk.metadata or {}
    parent_chunk_id = metadata.get("parent_chunk_id")
    parent_text = metadata.get("parent_text")
    if isinstance(parent_text, str) and parent_text and not _is_truncated_parent_text(parent_text):
        return parent_text, str(parent_chunk_id or hit.chunk.chunk_id)
    return hit.chunk.text, hit.chunk.chunk_id


def build_contexts(
    hits: list[SearchHit],
    limit: int,
    quotas: dict[str, int] | None = None,
    bucket_order: list[str] | None = None,
    query: str | None = None,
) -> list[dict[str, object]]:
    contexts = []
    seen_keys: set[str] = set()
    for hit in hits:
        metadata = hit.chunk.metadata or {}
        parent_chunk_id = metadata.get("parent_chunk_id")
        context_text, dedupe_key = _context_text_and_dedupe_key(hit)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        section_path = metadata.get("section_path")
        section_path_text = (
            " > ".join(section_path)
            if isinstance(section_path, list)
            else metadata.get("section_path_text")
        )
        context_entry: dict[str, object] = {
            "chunk_id": hit.chunk.chunk_id,
            "_dedupe_key": dedupe_key,
            "text": context_text,
            "source": hit.chunk.source_path,
            "title": section_path_text or hit.chunk.title,
            "score": round(hit.score, 6),
            "wiki_kind": metadata.get("wiki_kind"),
            "wiki_status": metadata.get("wiki_status"),
            "original_source_path": metadata.get("original_source_path"),
            "section_path": section_path,
            "parent_chunk_id": parent_chunk_id,
            "doc_type": metadata.get("doc_type"),
            "effective_date": metadata.get("effective_date"),
            "version": metadata.get("version"),
            "source_key": metadata.get("source_key"),
            "freshness_tier": metadata.get("freshness_tier"),
            "is_latest_version": metadata.get("is_latest_version"),
            "freshness_rank": metadata.get("freshness_rank"),
            "chunk_kind": metadata.get("chunk_kind"),
            "supporting_chunk_id": hit.chunk.chunk_id,
            "modality": hit.chunk.modality,
            "media_uri": hit.chunk.media_uri,
            "mime_type": hit.chunk.mime_type,
        }
        context_entry["evidence_role"] = _evidence_role(context_entry)
        contexts.append(context_entry)
    if quotas is None:
        selected = _promote_query_coverage(contexts[:limit], contexts, query)
        return _with_citation_labels(_order_contexts_by_evidence(selected))

    ordered_buckets = bucket_order or DEFAULT_BUCKET_ORDER
    grouped: dict[str, list[dict[str, object]]] = {bucket: [] for bucket in ordered_buckets}
    extras: list[dict[str, object]] = []
    for context in contexts:
        bucket = _context_bucket(context)
        if bucket in grouped:
            grouped[bucket].append(context)
        else:
            extras.append(context)

    selected: list[dict[str, object]] = []
    positions = {bucket: 0 for bucket in ordered_buckets}
    selected_per_bucket = {bucket: 0 for bucket in ordered_buckets}

    while len(selected) < limit:
        progressed = False
        for bucket in ordered_buckets:
            if selected_per_bucket[bucket] >= quotas.get(bucket, limit):
                continue
            bucket_items = grouped[bucket]
            if positions[bucket] >= len(bucket_items):
                continue
            selected.append(bucket_items[positions[bucket]])
            positions[bucket] += 1
            selected_per_bucket[bucket] += 1
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break

    if len(selected) < limit:
        selected_keys = {_dedupe_key(item) for item in selected}
        for context in [*contexts, *extras]:
            dedupe_key = _dedupe_key(context)
            if dedupe_key in selected_keys:
                continue
            selected.append(context)
            selected_keys.add(dedupe_key)
            if len(selected) >= limit:
                break

    selected = _promote_query_coverage(selected[:limit], contexts, query)
    return _with_citation_labels(_order_contexts_by_evidence(selected))
