from __future__ import annotations

from enum import Enum
from typing import Iterable

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    ROOT = "root"
    SECTION = "section"
    CLAUSE = "clause"
    DEFINITION = "definition"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    IMAGE = "image"
    PAGE_REGION = "page_region"
    LIST = "list"


class BoundingBox(BaseModel):
    page_number: int
    x0: float
    y0: float
    x1: float
    y1: float
    coord_system: str = "page"


class NodeProvenance(BaseModel):
    source_document_id: str
    page_number: int | None = None
    hierarchy_path: list[str] = Field(default_factory=list)
    bounding_box: BoundingBox | None = None
    source_ref: str | None = None


class TableCell(BaseModel):
    row: int
    col: int
    text: str
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False
    bounding_box: BoundingBox | None = None


class TablePayload(BaseModel):
    rows: int
    cols: int
    cells: list[TableCell] = Field(default_factory=list)
    header_map: dict[str, list[int]] = Field(default_factory=dict)
    caption: str | None = None
    narrative: str | None = None


class GraphEntity(BaseModel):
    entity_id: str
    name: str
    entity_type: str = "concept"
    source_node_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)


class GraphRelation(BaseModel):
    relation_id: str
    source_id: str
    target_id: str
    relation_type: str
    source_node_id: str | None = None
    confidence: float = 1.0
    metadata: dict[str, object] = Field(default_factory=dict)


class KnowledgeGraph(BaseModel):
    entities: list[GraphEntity] = Field(default_factory=list)
    relations: list[GraphRelation] = Field(default_factory=list)


class DocumentNode(BaseModel):
    node_id: str
    doc_id: str
    kb_id: str
    node_type: NodeType
    text: str = ""
    title: str | None = None
    parent_id: str | None = None
    children: list["DocumentNode"] = Field(default_factory=list)
    provenance: NodeProvenance
    table: TablePayload | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return not self.children


class StructuredDocument(BaseModel):
    doc_id: str
    kb_id: str
    source_path: str
    title: str
    root: DocumentNode
    graph: KnowledgeGraph = Field(default_factory=KnowledgeGraph)
    metadata: dict[str, object] = Field(default_factory=dict)

    def iter_nodes(self) -> Iterable[DocumentNode]:
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

    def iter_leaves(self) -> Iterable[DocumentNode]:
        for node in self.iter_nodes():
            if node.is_leaf and node.node_type != NodeType.ROOT:
                yield node
