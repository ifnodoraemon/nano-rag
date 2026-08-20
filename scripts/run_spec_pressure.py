from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import AppContainer
from app.schemas.chat import ChatRequest
from scripts.live_smoke import load_env_file, normalize_provider_env


CORPUS = [
    {
        "name": "rfc9110_http_semantics.txt",
        "url": "https://www.rfc-editor.org/rfc/rfc9110.txt",
        "kind": "ietf-rfc",
    },
    {
        "name": "rfc9111_http_caching.txt",
        "url": "https://www.rfc-editor.org/rfc/rfc9111.txt",
        "kind": "ietf-rfc",
    },
    {
        "name": "rfc8446_tls13.txt",
        "url": "https://www.rfc-editor.org/rfc/rfc8446.txt",
        "kind": "ietf-rfc",
    },
    {
        "name": "rfc9000_quic_transport.txt",
        "url": "https://www.rfc-editor.org/rfc/rfc9000.txt",
        "kind": "ietf-rfc",
    },
    {
        "name": "wcag22.html",
        "url": "https://www.w3.org/TR/WCAG22/",
        "kind": "w3c-recommendation",
    },
    {
        "name": "openapi_3_1_1.md",
        "url": "https://raw.githubusercontent.com/OAI/OpenAPI-Specification/main/versions/3.1.1.md",
        "kind": "openapi-spec",
    },
    {
        "name": "nist_sp_800_207_zero_trust.pdf",
        "url": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-207.pdf",
        "kind": "nist-sp",
    },
    {
        "name": "nist_sp_800_63b_digital_identity.pdf",
        "url": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63b.pdf",
        "kind": "nist-sp",
    },
]

LOCAL_TABLE_SAMPLE = """# 工程验收矩阵样例

## 1 范围

本样例用于批量压测表格结构解析、表头映射、逻辑坐标和溯源链路，不代表任何真实国家标准。

## 2 验收项目

| 项目编号 | 子系统 | 指标 | 合格条件 | 复验周期 |
| --- | --- | --- | --- | --- |
| A-01 | 供配电 | 双路输入切换 | 切换时间不超过 2 秒，业务无中断 | 12 个月 |
| A-02 | 消防联动 | 告警转发 | 平台在 10 秒内收到联动事件 | 6 个月 |
| A-03 | 机房环境 | 温湿度采集 | 采集间隔不超过 60 秒 | 3 个月 |

## 3 处置要求

当同一子系统连续两次复验不合格时，应生成整改闭环记录并关联原始检测报告。
"""

LOCAL_TEXT_SAMPLE = """1 总则

本地样例用于覆盖纯文本规范的编号标题、长段落、交叉引用和实体关系抽取。

1.1 术语

控制项是指可被验证、审计和追踪的管理或技术要求。证据项是指证明控制项已执行的记录。

2 控制要求

控制项 C-01 属于访问控制域。控制项 C-01 应引用证据项 E-01 和 E-02。
控制项 C-02 属于变更管理域。控制项 C-02 应在发布窗口结束后 24 小时内完成复核。
"""

QUERIES = [
    "RFC 9110 中哪些 HTTP 方法被定义为 safe，safe 的含义是什么？",
    "RFC 9111 如何定义缓存的 stale response，什么情况下可以复用？",
    "RFC 8446 中 TLS 1.3 key schedule 的目的是什么？",
    "RFC 9000 中 QUIC connection ID 的作用是什么？",
    "WCAG 2.2 对 Focus Appearance 的核心要求是什么？",
    "OpenAPI 3.1.1 中 Paths Object 和 Operation Object 的关系是什么？",
    "NIST SP 800-207 如何定义 Zero Trust Architecture？",
    "NIST SP 800-63B 对 memorized secrets 有哪些关键要求？",
    "工程验收矩阵中 A-02 的子系统、合格条件和复验周期是什么？",
    "控制项 C-02 属于哪个域，要求在什么时间内完成复核？",
]


def download_file(url: str, path: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "nano-rag-spec-pressure/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        path.write_bytes(response.read())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run production-like spec pressure test.")
    parser.add_argument("--corpus-dir", default="data/raw/spec_pressure")
    parser.add_argument("--report-dir", default="data/reports/pressure")
    parser.add_argument("--kb-id", default="spec-pressure")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--query-concurrency", type=int, default=2)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-pdf", action="store_true")
    parser.add_argument("--limit-docs", type=int)
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Only ingest corpus records whose file name contains this substring. Can be repeated.",
    )
    parser.add_argument("--skip-queries", action="store_true")
    parser.add_argument("--graph-backend", choices=["neo4j", "artifact", "none"], default="neo4j")
    return parser.parse_args()


def configure_runtime(args: argparse.Namespace, corpus_dir: Path) -> None:
    load_env_file(ROOT / ".env")
    for key in (
        "MODEL_GATEWAY_API_KEY",
        "GENERATION_API_KEY",
        "DOCUMENT_PARSER_API_KEY",
        "MODEL_GATEWAY_BASE_URL",
        "GENERATION_API_BASE_URL",
        "DOCUMENT_PARSER_API_BASE_URL",
        "GENERATION_MODEL_ALIAS",
        "DOCUMENT_PARSER_MODEL",
    ):
        value = os.getenv(f"COMPOSE_{key}", "")
        if value:
            os.environ.setdefault(key, value)
    normalize_provider_env()
    os.environ["MODEL_GATEWAY_MODE"] = "live"
    os.environ.setdefault("RAG_WIKI_ENABLED", "true")
    os.environ["RAG_GRAPH_BACKEND"] = "artifact" if args.graph_backend == "none" else args.graph_backend
    os.environ["RAG_INGEST_ALLOWED_DIRS"] = str(corpus_dir.resolve())
    os.environ.setdefault("RAG_AUTH_DISABLED", "true")
    os.environ.setdefault("DISABLE_RERANK", "true")
    if args.graph_backend == "neo4j":
        os.environ.setdefault("NEO4J_URI", "bolt://127.0.0.1:7687")
        os.environ.setdefault("NEO4J_USER", "neo4j")
        os.environ.setdefault("NEO4J_PASSWORD", "nano-rag-graph")


def prepare_corpus(corpus_dir: Path, *, download: bool, force: bool, skip_pdf: bool) -> list[dict[str, Any]]:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "engineering_acceptance_matrix.md").write_text(
        LOCAL_TABLE_SAMPLE,
        encoding="utf-8",
    )
    (corpus_dir / "control_requirements.txt").write_text(
        LOCAL_TEXT_SAMPLE,
        encoding="utf-8",
    )

    records: list[dict[str, Any]] = []
    for item in CORPUS:
        path = corpus_dir / str(item["name"])
        if skip_pdf and path.suffix.lower() == ".pdf":
            continue
        record = {**item}
        if download and (force or not path.exists()):
            started = time.perf_counter()
            print(f"download_start name={path.name}", flush=True)
            try:
                download_file(str(item["url"]), path)
                record["download_seconds"] = round(time.perf_counter() - started, 3)
                print(
                    f"download_ok name={path.name} seconds={record['download_seconds']}",
                    flush=True,
                )
            except Exception as exc:
                record.update(
                    {
                        "path": str(path),
                        "extension": path.suffix.lower(),
                        "size_bytes": 0,
                        "download_error_type": exc.__class__.__name__,
                        "download_error": str(exc)[:500],
                    }
                )
                records.append(record)
                print(
                    f"download_failed name={path.name} error={record['download_error_type']}",
                    flush=True,
                )
                continue
        if path.exists():
            stat = path.stat()
            records.append(
                {
                    **record,
                    "path": str(path),
                    "extension": path.suffix.lower(),
                    "size_bytes": stat.st_size,
                }
            )
    for path in sorted(corpus_dir.glob("*")):
        if path.is_file() and not any(record["path"] == str(path) for record in records):
            records.append(
                {
                    "name": path.name,
                    "url": None,
                    "kind": "local-sample",
                    "path": str(path),
                    "extension": path.suffix.lower(),
                    "size_bytes": path.stat().st_size,
                }
            )
    return records


async def run_queries(
    container: AppContainer,
    *,
    kb_id: str,
    top_k: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def one(query: str) -> dict[str, Any]:
        async with semaphore:
            started = time.perf_counter()
            print(f"query_start query={query[:80]}", flush=True)
            try:
                response = await container.chat_pipeline.run(
                    ChatRequest(query=query, kb_id=kb_id, top_k=top_k)
                )
                seconds = round(time.perf_counter() - started, 3)
                print(
                    f"query_ok contexts={len(response.contexts)} citations={len(response.citations)} seconds={seconds}",
                    flush=True,
                )
                return {
                    "query": query,
                    "ok": True,
                    "seconds": seconds,
                    "contexts": len(response.contexts),
                    "citations": len(response.citations),
                    "trace_id": response.trace_id,
                    "answer_preview": response.answer.replace("\n", " ")[:500],
                    "sources": sorted({citation.source for citation in response.citations}),
                }
            except Exception as exc:
                seconds = round(time.perf_counter() - started, 3)
                print(
                    f"query_failed error={exc.__class__.__name__} seconds={seconds}",
                    flush=True,
                )
                return {
                    "query": query,
                    "ok": False,
                    "seconds": seconds,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc)[:1000],
                }

    return await asyncio.gather(*(one(query) for query in QUERIES))


async def ingest_records(
    container: AppContainer,
    records: list[dict[str, Any]],
    *,
    kb_id: str,
) -> dict[str, Any]:
    doc_results: list[dict[str, Any]] = []
    documents = 0
    chunks = 0
    started = time.perf_counter()
    for record in records:
        path = str(record["path"])
        if record.get("download_error"):
            doc_results.append(
                {
                    "path": path,
                    "ok": False,
                    "seconds": 0.0,
                    "error_type": str(record["download_error_type"]),
                    "error": str(record["download_error"]),
                }
            )
            continue
        doc_started = time.perf_counter()
        print(f"ingest_start path={Path(path).name}", flush=True)
        try:
            ingest = await container.ingestion_pipeline.run(path, kb_id=kb_id)
            documents += ingest.documents
            chunks += ingest.chunks
            seconds = round(time.perf_counter() - doc_started, 3)
            doc_results.append(
                {
                    "path": path,
                    "ok": True,
                    "documents": ingest.documents,
                    "chunks": ingest.chunks,
                    "seconds": seconds,
                }
            )
            print(
                f"ingest_ok path={Path(path).name} chunks={ingest.chunks} seconds={seconds}",
                flush=True,
            )
        except Exception as exc:
            seconds = round(time.perf_counter() - doc_started, 3)
            doc_results.append(
                {
                    "path": path,
                    "ok": False,
                    "seconds": seconds,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc)[:1000],
                }
            )
            print(
                f"ingest_failed path={Path(path).name} error={exc.__class__.__name__} seconds={seconds}",
                flush=True,
            )
    return {
        "documents": documents,
        "chunks": chunks,
        "seconds": round(time.perf_counter() - started, 3),
        "files": doc_results,
    }


def summarize_docs(corpus_records: list[dict[str, Any]]) -> dict[str, Any]:
    by_ext = Counter(str(record["extension"]) for record in corpus_records)
    by_kind = Counter(str(record["kind"]) for record in corpus_records)
    return {
        "documents": len(corpus_records),
        "bytes": sum(int(record["size_bytes"]) for record in corpus_records),
        "by_extension": dict(sorted(by_ext.items())),
        "by_kind": dict(sorted(by_kind.items())),
    }


def filter_records(records: list[dict[str, Any]], includes: list[str]) -> list[dict[str, Any]]:
    needles = [item.casefold() for item in includes if item.strip()]
    if not needles:
        return records
    return [
        record
        for record in records
        if any(needle in str(record["name"]).casefold() for needle in needles)
    ]


async def main_async() -> int:
    args = parse_args()
    corpus_dir = (ROOT / args.corpus_dir).resolve()
    report_dir = (ROOT / args.report_dir).resolve()
    configure_runtime(args, corpus_dir)
    corpus_records = prepare_corpus(
        corpus_dir,
        download=args.download,
        force=args.force_download,
        skip_pdf=args.skip_pdf,
    )
    corpus_records = filter_records(corpus_records, args.include)
    if args.limit_docs:
        corpus_records = corpus_records[: args.limit_docs]

    container: AppContainer | None = None
    started = time.perf_counter()
    try:
        container = AppContainer.from_env()
        ingest = await ingest_records(container, corpus_records, kb_id=args.kb_id)
        query_results = (
            []
            if args.skip_queries
            else await run_queries(
                container,
                kb_id=args.kb_id,
                top_k=args.top_k,
                concurrency=args.query_concurrency,
            )
        )
        ingest_failed = sum(1 for item in ingest["files"] if not item["ok"])
        query_failed = sum(1 for item in query_results if not item["ok"])
        report = {
            "status": "ok" if ingest_failed == 0 and query_failed == 0 else "partial",
            "created_at": int(time.time()),
            "runtime": {
                "discovery": container.wiki_searcher.stats() if container.wiki_searcher else {},
                "graph_backend": args.graph_backend,
                "kb_id": args.kb_id,
                "top_k": args.top_k,
                "query_concurrency": args.query_concurrency,
            },
            "corpus": summarize_docs(corpus_records),
            "corpus_records": corpus_records,
            "ingest": ingest,
            "queries": query_results,
            "summary": {
                "total_seconds": round(time.perf_counter() - started, 3),
                "ingest_failed": ingest_failed,
                "query_ok": sum(1 for item in query_results if item["ok"]),
                "query_failed": query_failed,
                "avg_query_seconds": round(
                    sum(float(item["seconds"]) for item in query_results) / max(len(query_results), 1),
                    3,
                ),
            },
        }
        report_dir.mkdir(parents=True, exist_ok=True)
        output = report_dir / f"spec_pressure_{int(time.time())}.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
        print(str(output), flush=True)
        return 0 if report["status"] == "ok" else 1
    finally:
        if container is not None:
            await container.close()


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
