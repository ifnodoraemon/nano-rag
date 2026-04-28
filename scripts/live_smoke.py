import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import AppContainer
from app.schemas.chat import ChatRequest


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def configured(name: str) -> str:
    value = os.getenv(name, "")
    return "set" if value and value != "change-me" else "missing"


async def run_smoke(args: argparse.Namespace) -> int:
    load_env_file(ROOT / ".env")
    if args.api_key_stdin:
        api_key = sys.stdin.readline().strip()
        if api_key:
            os.environ["MODEL_GATEWAY_API_KEY"] = api_key
    if args.generation_model:
        os.environ["GENERATION_MODEL_ALIAS"] = args.generation_model
    if args.embedding_model:
        os.environ["EMBEDDING_MODEL_ALIAS"] = args.embedding_model
    if args.rerank_model:
        os.environ["RERANK_MODEL_ALIAS"] = args.rerank_model
    os.environ["MODEL_GATEWAY_MODE"] = "live"
    if not args.use_config_vectorstore:
        os.environ["VECTORSTORE_BACKEND"] = "memory"
    os.environ["RAG_INGEST_ALLOWED_DIRS"] = str((ROOT / "data/raw").resolve())
    os.environ.setdefault("RAG_AUTH_DISABLED", "true")

    container = None
    try:
        container = AppContainer.from_env()
        print(f"MODEL_GATEWAY_MODE={container.config.gateway_mode}", flush=True)
        print(f"MODEL_GATEWAY_BASE_URL={container.config.gateway_base_url}", flush=True)
        print(f"MODEL_GATEWAY_API_KEY={configured('MODEL_GATEWAY_API_KEY')}", flush=True)
        print(f"GENERATION_MODEL_ALIAS={container.generation_client.alias}", flush=True)
        print(f"EMBEDDING_MODEL_ALIAS={container.embedding_client.alias}", flush=True)
        print(f"RERANK_MODEL_ALIAS={container.rerank_client.alias}", flush=True)
        print(f"rerank_enabled={container.config.rerank_enabled}", flush=True)
        print(
            f"vectorstore={container.repository.stats().get('backend', 'unknown')}",
            flush=True,
        )

        vectors = await container.embedding_client.embed_texts(
            ["差旅报销多久内提交？"]
        )
        print(f"embedding_ok=true count={len(vectors)} dim={len(vectors[0])}", flush=True)

        generated = await container.generation_client.generate(
            [
                {"role": "system", "content": "You answer briefly."},
                {"role": "user", "content": "Return exactly: nano-rag-live-ok"},
            ]
        )
        content = str(generated.get("content", "")).replace("\n", " ")[:160]
        print(f"generation_ok={bool(content)} sample={content}", flush=True)

        if container.config.rerank_enabled:
            ranked = await container.rerank_client.rerank(
                "差旅报销多久内提交？",
                [
                    "病假超过 3 天需要提供医院证明。",
                    "员工应在出差结束后 15 个自然日内提交差旅报销申请。",
                ],
                2,
            )
            top_index = ranked[0].index if ranked else None
            print(f"rerank_ok={bool(ranked)} top_index={top_index}", flush=True)
        else:
            print("rerank_skipped=disabled_by_config", flush=True)

        ingest = await container.ingestion_pipeline.run(
            str(ROOT / "data/raw/employee_handbook.md"),
            kb_id="default",
            tenant_id=args.tenant_id,
        )
        print(
            f"ingest_ok=true documents={ingest.documents} chunks={ingest.chunks}",
            flush=True,
        )

        chat = await container.chat_pipeline.run(
            ChatRequest(
                query="差旅报销多久内提交？",
                kb_id="default",
                tenant_id=args.tenant_id,
                top_k=6,
            )
        )
        answer = chat.answer.replace("\n", " ")[:240]
        print(
            f"chat_ok={bool(chat.answer)} contexts={len(chat.contexts)} "
            f"trace_id={chat.trace_id} answer={answer}",
            flush=True,
        )
        return 0
    except Exception as exc:
        message = str(exc).replace("\n", " ")[:800]
        print(f"smoke_failed={exc.__class__.__name__}: {message}", flush=True)
        return 1
    finally:
        if container is not None:
            await container.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live model gateway smoke test.")
    parser.add_argument("--tenant-id", default="live-smoke")
    parser.add_argument("--generation-model")
    parser.add_argument("--embedding-model")
    parser.add_argument("--rerank-model")
    parser.add_argument("--api-key-stdin", action="store_true")
    parser.add_argument(
        "--use-config-vectorstore",
        action="store_true",
        help="Use VECTORSTORE_BACKEND from .env instead of forcing memory.",
    )
    args = parser.parse_args()
    return asyncio.run(run_smoke(args))


if __name__ == "__main__":
    raise SystemExit(main())
