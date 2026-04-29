import pytest

from app.schemas.chunk import Chunk
from app.schemas.document import Document
from app.vectorstore.repository import InMemoryVectorRepository, MilvusVectorRepository


def test_in_memory_repository_delete_by_source_removes_matching_documents_and_chunks() -> None:
    repository = InMemoryVectorRepository()
    source_path = "data/raw/employee_handbook.md"
    other_source = "data/raw/other.md"
    repository.upsert(
        Document(doc_id="doc-1", source_path=source_path, title="A", content="a", metadata={"kb_id": "default"}),
        [
            Chunk(
                chunk_id="doc-1:0",
                doc_id="doc-1",
                chunk_index=0,
                text="a",
                source_path=source_path,
                title="A",
                metadata={"kb_id": "default"},
            )
        ],
        [[1.0, 0.0]],
    )
    repository.upsert(
        Document(doc_id="doc-2", source_path=other_source, title="B", content="b", metadata={"kb_id": "default"}),
        [
            Chunk(
                chunk_id="doc-2:0",
                doc_id="doc-2",
                chunk_index=0,
                text="b",
                source_path=other_source,
                title="B",
                metadata={"kb_id": "default"},
            )
        ],
        [[0.0, 1.0]],
    )

    repository.delete_by_source(source_path, kb_id="default")

    stats = repository.stats()
    assert stats["documents"] == 1
    assert stats["chunks"] == 1
    remaining_hit = repository.search([0.0, 1.0], top_k=1, kb_id="default")[0]
    assert remaining_hit.chunk.source_path == other_source


def test_in_memory_repository_search_is_scoped_by_kb_and_tenant() -> None:
    repository = InMemoryVectorRepository()
    source_path = "data/raw/employee_handbook.md"
    repository.upsert(
        Document(doc_id="doc-1", source_path=source_path, title="A", content="a", metadata={"kb_id": "kb-a", "tenant_id": "t1"}),
        [
            Chunk(
                chunk_id="doc-1:0",
                doc_id="doc-1",
                chunk_index=0,
                text="alpha",
                source_path=source_path,
                title="A",
                metadata={"kb_id": "kb-a", "tenant_id": "t1"},
            )
        ],
        [[1.0, 0.0]],
    )
    repository.upsert(
        Document(doc_id="doc-2", source_path=source_path, title="A", content="b", metadata={"kb_id": "kb-b", "tenant_id": "t2"}),
        [
            Chunk(
                chunk_id="doc-2:0",
                doc_id="doc-2",
                chunk_index=0,
                text="beta",
                source_path=source_path,
                title="A",
                metadata={"kb_id": "kb-b", "tenant_id": "t2"},
            )
        ],
        [[0.0, 1.0]],
    )

    hits = repository.search([1.0, 0.0], top_k=5, kb_id="kb-a", tenant_id="t1")

    assert len(hits) == 1
    assert hits[0].chunk.chunk_id == "doc-1:0"


def test_in_memory_repository_search_supports_metadata_filters() -> None:
    repository = InMemoryVectorRepository()
    source_path = "data/raw/employee_handbook.md"
    repository.upsert(
        Document(doc_id="doc-1", source_path=source_path, title="Policy", content="a", metadata={"kb_id": "default"}),
        [
            Chunk(
                chunk_id="doc-1:0",
                doc_id="doc-1",
                chunk_index=0,
                text="policy text",
                source_path=source_path,
                title="Policy",
                metadata={"kb_id": "default", "doc_type": "policy", "effective_date": "2026-01-15"},
            )
        ],
        [[1.0, 0.0]],
    )
    repository.upsert(
        Document(doc_id="doc-2", source_path="data/raw/faq.md", title="FAQ", content="b", metadata={"kb_id": "default"}),
        [
            Chunk(
                chunk_id="doc-2:0",
                doc_id="doc-2",
                chunk_index=0,
                text="faq text",
                source_path="data/raw/faq.md",
                title="FAQ",
                metadata={"kb_id": "default", "doc_type": "faq", "effective_date": "2024-01-15"},
            )
        ],
        [[1.0, 0.0]],
    )

    hits = repository.search(
        [1.0, 0.0],
        top_k=5,
        kb_id="default",
        metadata_filters={"doc_types": ["policy"], "effective_date_from": "2026-01-01"},
    )

    assert [hit.chunk.chunk_id for hit in hits] == ["doc-1:0"]


def test_milvus_repository_refuses_to_drop_collection_on_dimension_mismatch(monkeypatch) -> None:
    import types

    fake_pymilvus = types.ModuleType("pymilvus")
    fake_pymilvus.DataType = type("DataType", (), {"VARCHAR": 1, "FLOAT_VECTOR": 2})
    monkeypatch.setitem(__import__("sys").modules, "pymilvus", fake_pymilvus)

    class FakeMilvusClient:
        def has_collection(self, collection_name):  # noqa: ANN001, ARG002
            return True

        def describe_collection(self, collection_name):  # noqa: ANN001, ARG002
            return {
                "fields": [
                    {
                        "name": "vector",
                        "params": {"dim": 768},
                    }
                ]
            }

    monkeypatch.setattr("app.vectorstore.repository.create_milvus_client", lambda: FakeMilvusClient())

    with pytest.raises(RuntimeError) as exc_info:
        MilvusVectorRepository(dimension=3072)

    assert "Refusing to drop the collection automatically" in str(exc_info.value)


def test_milvus_repository_new_collection_includes_native_hybrid_schema(monkeypatch) -> None:
    import types

    class FakeDataType:
        VARCHAR = "VARCHAR"
        FLOAT_VECTOR = "FLOAT_VECTOR"
        SPARSE_FLOAT_VECTOR = "SPARSE_FLOAT_VECTOR"

    class FakeFunctionType:
        BM25 = "BM25"

    class FakeFunction:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    fake_pymilvus = types.ModuleType("pymilvus")
    fake_pymilvus.DataType = FakeDataType
    fake_pymilvus.FunctionType = FakeFunctionType
    fake_pymilvus.Function = FakeFunction
    monkeypatch.setitem(__import__("sys").modules, "pymilvus", fake_pymilvus)

    class FakeSchema:
        def __init__(self) -> None:
            self.fields = []
            self.functions = []

        def add_field(self, **kwargs) -> None:
            self.fields.append(kwargs)

        def add_function(self, function) -> None:  # noqa: ANN001
            self.functions.append(function)

    class FakeIndexParams:
        def __init__(self) -> None:
            self.indexes = []

        def add_index(self, **kwargs) -> None:
            self.indexes.append(kwargs)

    class FakeMilvusClient:
        def __init__(self) -> None:
            self.schema = None
            self.index_params = None

        def has_collection(self, collection_name):  # noqa: ANN001, ARG002
            return False

        def create_schema(self, **kwargs):  # noqa: ANN001, ARG002
            self.schema = FakeSchema()
            return self.schema

        def prepare_index_params(self):
            self.index_params = FakeIndexParams()
            return self.index_params

        def create_collection(self, collection_name, schema, index_params):  # noqa: ANN001, ARG002
            self.schema = schema
            self.index_params = index_params

    fake_client = FakeMilvusClient()
    monkeypatch.setattr("app.vectorstore.repository.create_milvus_client", lambda: fake_client)

    MilvusVectorRepository(dimension=3072)

    fields = {field["field_name"]: field for field in fake_client.schema.fields}
    assert fields["text"]["enable_analyzer"] is True
    assert fields["sparse"]["datatype"] == FakeDataType.SPARSE_FLOAT_VECTOR
    assert fake_client.schema.functions[0].kwargs["function_type"] == FakeFunctionType.BM25
    sparse_index = next(
        index for index in fake_client.index_params.indexes if index["field_name"] == "sparse"
    )
    assert sparse_index["metric_type"] == "BM25"


def test_milvus_repository_search_specifies_dense_vector_field(monkeypatch) -> None:
    import types

    fake_pymilvus = types.ModuleType("pymilvus")
    fake_pymilvus.DataType = type(
        "DataType",
        (),
        {
            "VARCHAR": "VARCHAR",
            "FLOAT_VECTOR": "FLOAT_VECTOR",
            "SPARSE_FLOAT_VECTOR": "SPARSE_FLOAT_VECTOR",
        },
    )
    monkeypatch.setitem(__import__("sys").modules, "pymilvus", fake_pymilvus)

    class FakeMilvusClient:
        def __init__(self) -> None:
            self.search_kwargs = None

        def has_collection(self, collection_name):  # noqa: ANN001, ARG002
            return True

        def describe_collection(self, collection_name):  # noqa: ANN001, ARG002
            return {
                "fields": [
                    {"name": "vector", "params": {"dim": 1536}},
                    {"name": "text"},
                    {"name": "sparse"},
                ]
            }

        def search(self, **kwargs):  # noqa: ANN001
            self.search_kwargs = kwargs
            return [[
                {
                    "distance": 0.99,
                    "entity": {
                        "chunk_id": "doc-1:0",
                        "doc_id": "doc-1",
                        "source": "data/raw/a.md",
                        "title": "A",
                        "chunk_index": 0,
                        "text": "hello",
                        "metadata_json": {"kb_id": "default"},
                        "modality": "text",
                        "media_uri": "",
                        "mime_type": "",
                    },
                }
            ]]

    fake_client = FakeMilvusClient()
    monkeypatch.setattr("app.vectorstore.repository.create_milvus_client", lambda: fake_client)

    repository = MilvusVectorRepository(dimension=1536)
    hits = repository.search([0.1] * 1536, top_k=1)

    assert hits[0].chunk.chunk_id == "doc-1:0"
    assert fake_client.search_kwargs["anns_field"] == "vector"
