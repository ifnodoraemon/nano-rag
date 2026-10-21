"""scripts/init_env.py — the operator's single configuration entry point.

The script writes real secrets, so the parts worth pinning are the ones that can
silently produce a broken stack: value preservation across re-runs, legacy-name
migration, and the checks that catch a misconfigured file (identical keys, a
multi-key proxy value).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "init_env", ROOT / "scripts" / "init_env.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def init_env(tmp_path, monkeypatch):
    module = _load_module()
    env_path = tmp_path / "docker" / ".env"
    monkeypatch.setattr(module, "ENV_PATH", env_path)
    return module, env_path


def _read(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def test_write_generates_distinct_secrets(init_env) -> None:
    module, env_path = init_env
    assert module.write(force=False) == 0

    values = _read(env_path)
    assert values["RAG_API_KEYS"]
    assert values["RAG_ADMIN_API_KEYS"]
    assert values["RAG_API_KEYS"] != values["RAG_ADMIN_API_KEYS"]
    assert len(values["RAG_API_KEYS"]) == 64  # token_hex(32)
    assert values["POSTGRES_PASSWORD"]
    # The browser-facing proxy key is derived, not a fourth secret to maintain.
    assert "RAG_PROXY_API_KEY" not in values


def test_rerun_preserves_operator_provider_keys(init_env) -> None:
    module, env_path = init_env
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(
        "GENERATION_API_KEY=my-real-key\n"
        "GENERATION_MODEL_ALIAS=qwen-plus\n"
        "RAG_API_KEYS=existing-business\n"
        "RAG_ADMIN_API_KEYS=existing-admin\n"
        "POSTGRES_PASSWORD=existing-db\n",
        encoding="utf-8",
    )

    module.write(force=False)

    values = _read(env_path)
    assert values["GENERATION_API_KEY"] == "my-real-key"
    assert values["GENERATION_MODEL_ALIAS"] == "qwen-plus"
    assert values["RAG_API_KEYS"] == "existing-business"
    assert values["POSTGRES_PASSWORD"] == "existing-db"


def test_legacy_compose_names_are_migrated_not_lost(init_env) -> None:
    module, env_path = init_env
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(
        "COMPOSE_GENERATION_API_KEY=legacy-key\n"
        "COMPOSE_DOCUMENT_PARSER_API_KEY=legacy-parser\n"
        "COMPOSE_GENERATION_MODEL_ALIAS=qwen-plus\n",
        encoding="utf-8",
    )

    module.write(force=False)

    values = _read(env_path)
    # Values survive, names become canonical.
    assert values["GENERATION_API_KEY"] == "legacy-key"
    assert values["DOCUMENT_PARSER_API_KEY"] == "legacy-parser"
    assert values["GENERATION_MODEL_ALIAS"] == "qwen-plus"
    assert not any(key.startswith("COMPOSE_") for key in values)
    # Generated secrets are added by the migration.
    assert values["RAG_ADMIN_API_KEYS"]


def test_force_rotates_secrets(init_env) -> None:
    module, env_path = init_env
    module.write(force=False)
    before = _read(env_path)
    module.write(force=True)
    after = _read(env_path)

    assert before["RAG_API_KEYS"] != after["RAG_API_KEYS"]
    # A backup of the previous file is kept.
    assert (env_path.parent / f"{env_path.name}.bak").exists()


def test_check_reports_missing_provider_key(init_env, capsys) -> None:
    module, env_path = init_env
    module.write(force=False)  # secrets present, provider keys empty

    assert module.check() == 1
    output = capsys.readouterr().out
    assert "GENERATION_API_KEY" in output


def test_check_rejects_identical_business_and_admin_keys(init_env, capsys) -> None:
    module, env_path = init_env
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(
        "GENERATION_API_KEY=k\nDOCUMENT_PARSER_API_KEY=k\n"
        "RAG_API_KEYS=same\nRAG_ADMIN_API_KEYS=same\nPOSTGRES_PASSWORD=p\n",
        encoding="utf-8",
    )

    assert module.check() == 1
    assert "identical" in capsys.readouterr().out


def test_check_rejects_multiple_business_keys(init_env, capsys) -> None:
    # nginx injects exactly one key, so a comma list would silently fail to
    # authenticate browser traffic.
    module, env_path = init_env
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(
        "GENERATION_API_KEY=k\nDOCUMENT_PARSER_API_KEY=k\n"
        "RAG_API_KEYS=a,b\nRAG_ADMIN_API_KEYS=admin\nPOSTGRES_PASSWORD=p\n",
        encoding="utf-8",
    )

    assert module.check() == 1
    assert "single key" in capsys.readouterr().out


def test_check_passes_when_ready(init_env) -> None:
    module, env_path = init_env
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(
        "GENERATION_API_KEY=k\nDOCUMENT_PARSER_API_KEY=k\n"
        "RAG_API_KEYS=business\nRAG_ADMIN_API_KEYS=admin\nPOSTGRES_PASSWORD=p\n",
        encoding="utf-8",
    )

    assert module.check() == 0


def test_placeholder_values_do_not_count_as_supplied(init_env) -> None:
    module, env_path = init_env
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(
        "GENERATION_API_KEY=<your-key-here>\nDOCUMENT_PARSER_API_KEY=k\n"
        "RAG_API_KEYS=business\nRAG_ADMIN_API_KEYS=admin\nPOSTGRES_PASSWORD=p\n",
        encoding="utf-8",
    )

    assert module.check() == 1
