import argparse
import asyncio
import getpass
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("live_smoke")

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


def read_secret_from_stdin(prompt: str) -> str:
    if sys.stdin.isatty():
        return getpass.getpass(prompt).strip()
    return sys.stdin.readline().strip()


def normalize_provider_env() -> None:
    gateway_key = os.getenv("MODEL_GATEWAY_API_KEY", "")
    if gateway_key:
        os.environ.setdefault("GENERATION_API_KEY", gateway_key)
        os.environ.setdefault("DOCUMENT_PARSER_API_KEY", gateway_key)
    gateway_base_url = os.getenv("MODEL_GATEWAY_BASE_URL", "")
    if gateway_base_url:
        os.environ.setdefault("GENERATION_API_BASE_URL", gateway_base_url)
    # Do NOT default the parser endpoint: a configured DashScope/Qwen parser
    # must not be silently redirected to Google.
    if not os.getenv("DOCUMENT_PARSER_API_BASE_URL"):
        logger.warning(
            "DOCUMENT_PARSER_API_BASE_URL is unset; the parser capability will "
            "report as unconfigured instead of guessing a provider endpoint."
        )
    os.environ.setdefault("RAG_WIKI_ENABLED", "true")


async def run_smoke(args: argparse.Namespace) -> int:
    load_env_file(ROOT / ".env")
    if args.api_key_stdin:
        api_key = read_secret_from_stdin("MODEL_GATEWAY_API_KEY: ")
        if api_key:
            os.environ["MODEL_GATEWAY_API_KEY"] = api_key
    if args.generation_model:
        os.environ["GENERATION_MODEL_ALIAS"] = args.generation_model
    normalize_provider_env()
    os.environ["MODEL_GATEWAY_MODE"] = "live"
    os.environ["RAG_INGEST_ALLOWED_DIRS"] = str((ROOT / "data/raw").resolve())
    os.environ.setdefault("RAG_AUTH_DISABLED", "true")

    container = None
    try:
        container = AppContainer.from_env()
        print(f"MODEL_GATEWAY_MODE={container.config.gateway_mode}", flush=True)
        print(f"MODEL_GATEWAY_BASE_URL={container.config.gateway_base_url}", flush=True)
        print(f"MODEL_GATEWAY_API_KEY={configured('MODEL_GATEWAY_API_KEY')}", flush=True)
        print(f"GENERATION_MODEL_ALIAS={container.generation_client.alias}", flush=True)
        if container.wiki_searcher is not None:
            print(f"discovery={container.wiki_searcher.stats()}", flush=True)

        generated = await container.generation_client.generate(
            [
                {"role": "system", "content": "You answer briefly."},
                {"role": "user", "content": "Return exactly: nano-rag-live-ok"},
            ]
        )
        raw_content = str(generated.get("content", ""))
        content = raw_content.replace("\n", " ")[:160]
        # The prompt asks for a sentinel; "non-empty" alone would report success
        # for an error string or a refusal.
        generation_ok = "nano-rag-live-ok" in raw_content.casefold()
        print(f"generation_ok={generation_ok} sample={content}", flush=True)
        if not generation_ok:
            print(
                "generation check failed: the model did not return the "
                "'nano-rag-live-ok' sentinel",
                flush=True,
            )
            return 1


        ingest = await container.ingestion_pipeline.run(
            str(ROOT / "data/raw/employee_handbook.md"),
            kb_id="default",
        )
        ingest_ok = ingest.documents > 0 and ingest.chunks > 0
        print(
            f"ingest_ok={ingest_ok} documents={ingest.documents} chunks={ingest.chunks}",
            flush=True,
        )
        if not ingest_ok:
            print("ingest check failed: no documents/chunks were produced", flush=True)
            return 1

        chat = await container.chat_pipeline.run(
            ChatRequest(
                query="差旅报销多久内提交？",
                kb_id="default",
                top_k=6,
            )
        )
        answer = chat.answer.replace("\n", " ")[:240]
        chat_ok = bool(chat.answer.strip()) and bool(chat.contexts) and bool(chat.trace_id)
        print(
            f"chat_ok={chat_ok} contexts={len(chat.contexts)} "
            f"trace_id={chat.trace_id} answer={answer}",
            flush=True,
        )
        if not chat_ok:
            print(
                "chat check failed: an answer needs text, retrieved context and "
                "a trace id to be traceable",
                flush=True,
            )
            return 1
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
    parser.add_argument("--generation-model")
    parser.add_argument("--api-key-stdin", action="store_true")
    args = parser.parse_args()
    return asyncio.run(run_smoke(args))


if __name__ == "__main__":
    raise SystemExit(main())
