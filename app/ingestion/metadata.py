from __future__ import annotations

import re
from dataclasses import dataclass

MAX_PARENT_TEXT_CHARS = 1500

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
DATE_PATTERNS = (
    re.compile(r"((?:19|20)\d{2}-\d{1,2}-\d{1,2})"),
    re.compile(r"((?:19|20)\d{2}/\d{1,2}/\d{1,2})"),
    re.compile(r"((?:19|20)\d{2}年\d{1,2}月\d{1,2}日)"),
)
VERSION_PATTERNS = (
    re.compile(r"\b(v\d+(?:\.\d+)*)\b", re.IGNORECASE),
    re.compile(r"\b(version\s+\d+(?:\.\d+)*)\b", re.IGNORECASE),
    re.compile(r"(版本[:：]?\s*[A-Za-z0-9._-]+)"),
)
FIELD_PATTERNS = {
    "owner": (
        re.compile(r"(?:owner|负责人)[:：]\s*([^\n]+)", re.IGNORECASE),
        re.compile(r"(?:维护人|维护团队)[:：]\s*([^\n]+)"),
    ),
    "department": (
        re.compile(r"(?:department|部门)[:：]\s*([^\n]+)", re.IGNORECASE),
        re.compile(r"(?:所属部门)[:：]\s*([^\n]+)"),
    ),
}
SOURCE_KEY_SUFFIX_RE = re.compile(
    r"(\b(?:v(?:ersion)?\s*\d+(?:\.\d+)*)\b|\b(?:19|20)\d{2}\b|\b(?:final|latest|draft)\b)",
    re.IGNORECASE,
)


@dataclass
class Section:
    title: str
    level: int
    path: list[str]
    text: str


def extract_document_metadata(
    source_path: str,
    title: str,
    text: str,
    kb_id: str = "default",
    tenant_id: str | None = None,
) -> dict[str, object]:
    headings = extract_headings(text)
    metadata: dict[str, object] = {
        "kb_id": kb_id,
        "tenant_id": tenant_id,
        "doc_type": infer_doc_type(source_path, title, text),
        "source_key": build_source_key(source_path, title),
        "headings": headings,
        "section_count": len(split_sections(text, title)),
    }
    effective_date = _extract_first_match(text, DATE_PATTERNS)
    version = _extract_first_match(text, VERSION_PATTERNS)
    owner = _extract_field(text, "owner")
    department = _extract_field(text, "department")
    if effective_date:
        metadata["effective_date"] = effective_date
    if version:
        metadata["version"] = version
    if owner:
        metadata["owner"] = owner
    if department:
        metadata["department"] = department
    return metadata


def infer_doc_type(source_path: str, title: str, text: str) -> str:
    combined = f"{source_path}\n{title}\n{text[:500]}".lower()
    heuristics = (
        ("faq", ("faq", "q&a", "常见问题", "问答")),
        ("policy", ("policy", "政策", "制度", "办法", "规范")),
        ("handbook", ("handbook", "手册", "员工手册")),
        ("guide", ("guide", "guideline", "指南", "指引")),
        ("procedure", ("procedure", "流程", "操作步骤", "操作规程")),
        ("contract", ("contract", "agreement", "合同", "协议")),
        ("form", ("form", "template", "表单", "模板")),
    )
    for doc_type, markers in heuristics:
        if any(marker in combined for marker in markers):
            return doc_type
    return "document"


def build_source_key(source_path: str, title: str) -> str:
    candidate = title.strip() or source_path.rsplit("/", 1)[-1]
    candidate = re.sub(r"\.[A-Za-z0-9]+$", "", candidate)
    candidate = SOURCE_KEY_SUFFIX_RE.sub(" ", candidate)
    candidate = re.sub(r"[_\-/]+", " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip().lower()
    return candidate or "document"


def extract_headings(text: str) -> list[str]:
    headings: list[str] = []
    for line in text.splitlines():
        match = HEADING_RE.match(line.strip())
        if match:
            headings.append(match.group(2).strip())
    return headings


def split_sections(text: str, title: str) -> list[Section]:
    if not text.strip():
        return []

    sections: list[Section] = []
    stack: list[str] = []
    current_title = title
    current_level = 1
    current_path = [title]
    current_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if not body:
            return
        sections.append(
            Section(
                title=current_title,
                level=current_level,
                path=list(current_path),
                text=body,
            )
        )

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        heading = HEADING_RE.match(line.strip())
        if heading:
            flush()
            current_lines = [line.strip()]
            level = len(heading.group(1))
            heading_title = heading.group(2).strip()
            stack = stack[: max(level - 1, 0)]
            stack.append(heading_title)
            current_title = heading_title
            current_level = level
            current_path = [title, *stack]
            continue
        current_lines.append(line)

    flush()
    return sections or [Section(title=title, level=1, path=[title], text=text.strip())]


def build_section_metadata(
    base_metadata: dict[str, object] | None,
    section: Section,
    parent_chunk_id: str,
    child_chunk_count: int,
    child_chunk_index: int,
) -> dict[str, object]:
    metadata = dict(base_metadata or {})
    metadata.update(
        {
            "section_title": section.title,
            "section_level": section.level,
            "section_path": section.path,
            "section_path_text": " > ".join(section.path),
            "parent_chunk_id": parent_chunk_id,
            "parent_text": _preview(section.text, MAX_PARENT_TEXT_CHARS),
            "child_chunk_count": child_chunk_count,
            "child_chunk_index": child_chunk_index,
            "chunk_kind": "child",
        }
    )
    return metadata


def _extract_first_match(text: str, patterns: tuple[re.Pattern[str], ...]) -> str | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None


def _extract_field(text: str, field_name: str) -> str | None:
    for pattern in FIELD_PATTERNS[field_name]:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None


def _preview(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."
