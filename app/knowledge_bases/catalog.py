from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from time import time

from pydantic import BaseModel, Field


ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")


class KnowledgeBaseRecord(BaseModel):
    kb_id: str
    name: str
    description: str | None = None
    source: str = "local"
    external_ref: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: float
    updated_at: float


class _CatalogPayload(BaseModel):
    knowledge_bases: list[KnowledgeBaseRecord] = Field(default_factory=list)


class KnowledgeBaseCatalog:
    def __init__(self, path: Path, seed_kb_ids: set[str] | None = None) -> None:
        self.path = path
        self.seed_kb_ids = seed_kb_ids or {"default"}
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = self._load_unlocked()
            changed = self._ensure_defaults_unlocked(payload)
            if changed or not self.path.exists():
                self._save_unlocked(payload)

    def list(
        self, allowed_kb_ids: set[str] | None = None
    ) -> list[KnowledgeBaseRecord]:
        with self._lock:
            payload = self._load_unlocked()
        items = payload.knowledge_bases
        if allowed_kb_ids is not None:
            items = [item for item in items if item.kb_id in allowed_kb_ids]
        return sorted(items, key=lambda item: item.name.lower())

    def create(
        self,
        kb_id: str,
        name: str,
        description: str | None = None,
        source: str = "local",
        external_ref: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> KnowledgeBaseRecord:
        kb_id = self.validate_id(kb_id, "kb_id")
        name = self.validate_name(name)
        now = time()
        with self._lock:
            payload = self._load_unlocked()
            if any(item.kb_id == kb_id for item in payload.knowledge_bases):
                raise ValueError(f"knowledge base already exists: {kb_id}")
            record = KnowledgeBaseRecord(
                kb_id=kb_id,
                name=name,
                description=description,
                source=source or "local",
                external_ref=external_ref,
                metadata=metadata or {},
                created_at=now,
                updated_at=now,
            )
            payload.knowledge_bases.append(record)
            self._save_unlocked(payload)
            return record

    def get(self, kb_id: str) -> KnowledgeBaseRecord | None:
        with self._lock:
            payload = self._load_unlocked()
        for item in payload.knowledge_bases:
            if item.kb_id == kb_id:
                return item
        return None

    def exists(self, kb_id: str) -> bool:
        return self.get(kb_id) is not None

    def _load_unlocked(self) -> _CatalogPayload:
        if not self.path.exists():
            return _CatalogPayload()
        try:
            return _CatalogPayload.model_validate(
                json.loads(self.path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, ValueError):
            return _CatalogPayload()

    def _save_unlocked(self, payload: _CatalogPayload) -> None:
        tmp_path = self.path.with_suffix(".json.tmp")
        tmp_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
        os.replace(str(tmp_path), str(self.path))

    def _ensure_defaults_unlocked(self, payload: _CatalogPayload) -> bool:
        changed = False
        existing = {item.kb_id for item in payload.knowledge_bases}
        now = time()
        for kb_id in sorted(self.seed_kb_ids or {"default"}):
            normalized = self.validate_id(kb_id, "kb_id")
            if normalized in existing:
                continue
            name = "Default Knowledge Base" if normalized == "default" else normalized
            payload.knowledge_bases.append(
                KnowledgeBaseRecord(
                    kb_id=normalized,
                    name=name,
                    created_at=now,
                    updated_at=now,
                )
            )
            changed = True
        return changed

    @staticmethod
    def validate_id(value: str, field_name: str) -> str:
        normalized = str(value or "").strip()
        if not ID_PATTERN.match(normalized):
            raise ValueError(
                f"{field_name} must start with an alphanumeric character and contain only letters, numbers, dots, underscores, or hyphens"
            )
        return normalized

    @staticmethod
    def validate_name(value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("name must not be empty")
        return normalized[:160]
