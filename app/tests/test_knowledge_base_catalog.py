import pytest

from app.knowledge_bases.catalog import KnowledgeBaseCatalog


def test_catalog_creates_default_kb(tmp_path) -> None:
    catalog = KnowledgeBaseCatalog(tmp_path / "catalog.json")

    records = catalog.list()

    assert [record.kb_id for record in records] == ["default"]


def test_catalog_creates_and_persists_kb(tmp_path) -> None:
    path = tmp_path / "catalog.json"
    catalog = KnowledgeBaseCatalog(path)

    created = catalog.create(kb_id="policies", name="Policies")
    reloaded = KnowledgeBaseCatalog(path)

    assert created.kb_id == "policies"
    assert {record.kb_id for record in reloaded.list()} == {"default", "policies"}


def test_catalog_rejects_duplicate_kb(tmp_path) -> None:
    catalog = KnowledgeBaseCatalog(tmp_path / "catalog.json")
    catalog.create(kb_id="policies", name="Policies")

    with pytest.raises(ValueError):
        catalog.create(kb_id="policies", name="Policies")


def test_catalog_rejects_invalid_kb_id(tmp_path) -> None:
    catalog = KnowledgeBaseCatalog(tmp_path / "catalog.json")

    with pytest.raises(ValueError):
        catalog.create(kb_id="../bad", name="Bad")
