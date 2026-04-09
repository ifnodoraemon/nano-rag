from app.vectorstore.repository import SearchHit

DEFAULT_BUCKET_ORDER = ["topic", "raw", "source", "index"]
DEFAULT_EVIDENCE_ORDER = ["primary", "supporting", "conflicting"]


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
        labeled.append({**context, "citation_label": f"C{index}"})
    return labeled


def build_contexts(
    hits: list[SearchHit],
    limit: int,
    quotas: dict[str, int] | None = None,
    bucket_order: list[str] | None = None,
) -> list[dict[str, object]]:
    contexts = []
    seen_keys: set[str] = set()
    for hit in hits:
        metadata = hit.chunk.metadata or {}
        parent_chunk_id = metadata.get("parent_chunk_id")
        parent_text = metadata.get("parent_text")
        dedupe_key = str(parent_chunk_id or hit.chunk.chunk_id)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        section_path = metadata.get("section_path")
        section_path_text = (
            " > ".join(section_path)
            if isinstance(section_path, list)
            else metadata.get("section_path_text")
        )
        contexts.append(
            {
                "chunk_id": hit.chunk.chunk_id,
                "text": parent_text or hit.chunk.text,
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
            }
        )
        contexts[-1]["evidence_role"] = _evidence_role(contexts[-1])
    if quotas is None:
        return _with_citation_labels(_order_contexts_by_evidence(contexts[:limit]))

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
        selected_keys = {
            str(item.get("parent_chunk_id") or item.get("chunk_id")) for item in selected
        }
        for context in [*contexts, *extras]:
            dedupe_key = str(context.get("parent_chunk_id") or context.get("chunk_id"))
            if dedupe_key in selected_keys:
                continue
            selected.append(context)
            selected_keys.add(dedupe_key)
            if len(selected) >= limit:
                break

    return _with_citation_labels(_order_contexts_by_evidence(selected[:limit]))
