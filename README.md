# Nano RAG

Nano RAG 是一个真实数据优先的企业 RAG 工作台。后端负责文档注入、解析、分块、向量索引、检索、生成、追踪、评测和诊断；前端只展示后端已经支持的能力，不保存业务密钥，不内置 mock 数据，也不通过写死选项伪造状态。

![Nano RAG 运营台](./nanorag.png)

## 快速开始

```bash
docker compose -f docker/docker-compose.yml up -d --build
```

访问地址：

- 前端/API 统一入口：`http://127.0.0.1:3000`

默认只暴露前端一个 host 端口；后端、Milvus、Neo4j、Redis、MinIO 都只在 Docker 网络内通信。观测栈按需启动：

```bash
docker compose -f docker/docker-compose.yml --profile observability up -d langfuse-web langfuse-worker
```

- Langfuse：`http://127.0.0.1:3001`

如果只做本地 UI/接口验证，不需要 Milvus、Neo4j 和 Celery worker，可使用 lite 覆盖文件，只启动 `app` 和 `frontend` 两个容器：

```bash
docker compose -f docker/docker-compose.lite.yml up -d --build
```

lite 模式使用内存向量库、artifact 图扩展和 background ingest；正式规范压测仍建议使用默认生产栈。

浏览器端不需要配置 API key。前端 nginx 会为代理到后端的请求注入业务 key；本地 Docker 默认使用 `RAG_PROXY_API_KEY=nano-rag-local`，需要和后端 `RAG_API_KEYS` 中的一个值保持一致。

启动后优先检查真实运行状态：

```bash
curl -sS http://127.0.0.1:3000/health/detail
curl -sS http://127.0.0.1:3000/v1/rag/knowledge-bases
curl -sS http://127.0.0.1:3000/v1/rag/ingest/sources
```

## 项目理念

Nano RAG 的核心原则是：**真实输入、真实索引、真实模型、真实错误**。

- **不 mock**：运行时不使用 mock 网关，不把假数据展示成可用能力。
- **不回退**：模型、解析器、向量库和多模态 embedding 都必须显式配置；缺少配置或上游失败时直接暴露错误。
- **后端是事实来源**：知识库、可注入文件、文档列表、追踪、评测数据集、报告和诊断对象都来自后端接口。
- **前端不配置密钥**：浏览器通过上层业务后端、网关或 Docker 代理与后端连接；本地 Docker 代理会注入默认业务 API key。
- **知识库优先**：所有业务操作都挂在 `kb_id` 表示的知识库下，前端必须先选择后端返回的知识库。
- **证据优先**：回答必须带引用、上下文和 trace，问题排查必须能回到原始文档、分块和模型请求链路。

这意味着系统更愿意失败得明确，也不做静默降级。错误的 API key、不可用的解析器、缺失的 Milvus、空的 provider key，都应该被看见和修复。

## 当前能力

- 业务 API：`/v1/rag/chat`、`/v1/rag/retrieve`、`/v1/rag/ingest`、`/v1/rag/ingest/upload`、`/v1/rag/documents`、`/v1/rag/knowledge-bases`、`/v1/rag/ingest/sources`、`/v1/rag/feedback`、`/v1/rag/traces/{trace_id}`、`/v1/rag/benchmark/run`
- 运维 API：`/health/detail`、`/debug/storage`、`/debug/parsed/{doc_id}`、`/retrieve/debug`、`/traces`、`/replay/{trace_id}`
- 评测和诊断：`/eval/datasets`、`/eval/reports`、`/eval/run`、`/benchmark/reports`、`/diagnose/*`
- 向量库：Docker 默认使用 Milvus，collection 包含 dense vector、analyzer text、BM25 sparse 字段和原生 hybrid search
- 模型路径：生成、embedding、文档解析都直连显式配置的真实 provider；默认示例只覆盖 Gemini 和 Qwen
- 前端：React + Vite 源码位于 `frontend/` 子模块，Docker 构建为 nginx 静态站点并代理后端

## 目录

```text
nano-rag/
├─ app/              # FastAPI 后端
├─ configs/          # settings/models/prompts
├─ data/             # raw/eval/wiki 样例数据
├─ docker/           # Dockerfile、Compose、nginx 配置
├─ frontend/         # React + Vite 前端子模块
└─ scripts/          # eval/benchmark 等脚本
```

## Docker 配置

本地 Langfuse 初始化账号：

```bash
LANGFUSE_INIT_USER_EMAIL=admin@nano-rag.local
LANGFUSE_INIT_USER_PASSWORD=nano-rag-local-admin
LANGFUSE_PUBLIC_KEY=pk-lf-local
LANGFUSE_SECRET_KEY=sk-lf-local
```

这些 Langfuse 默认账号、项目 key、`NEXTAUTH_SECRET` 和 `ENCRYPTION_KEY` 只用于本地 Docker 环境。共享环境或生产环境必须显式覆盖，不能复用 compose 默认值。

Docker 默认值：

```bash
VECTORSTORE_BACKEND=milvus
MODEL_GATEWAY_MODE=live
GENERATION_API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
RAG_API_KEYS=nano-rag-local
RAG_DIAGNOSIS_ENABLED=true
RAG_EVAL_ENABLED=true
DOCUMENT_PARSER_ENABLED=true
LANGFUSE_UI_ENDPOINT=
LANGFUSE_OTEL_ENDPOINT=
```

后端启动时会检查 generation、embedding、document parser、Langfuse OTEL 等关键配置。缺少 `DOCUMENT_PARSER_API_KEY` 这类真实 provider 配置时，容器日志会输出 `Startup readiness` 警告，`/health/detail` 会继续显示对应能力不可用；前端只消费这些后端状态，不负责提示 Docker 配置方式。

## Provider 配置

真实模型调用需要配置 provider key。默认工程不再启动 Bifrost 或 LiteLLM，也不读取它们的密钥文件。只保留两套推荐配置：Gemini 或 Qwen。

模型 provider 抽象按能力拆分：

- `generation`：OpenAI-compatible chat completions，支持 Gemini、Qwen DashScope、Qwen vLLM。
- `embedding`：显式 provider adapter，支持 `gemini`、`dashscope`、`vllm`。
- `document_parser`：`gemini` 使用 Gemini Files API；`qwen` 使用 OpenAI-compatible chat completions，可指向 DashScope 或 vLLM。
- `rerank`：默认关闭；需要时显式配置 Qwen rerank endpoint 和 path。
- `trace`：只接 Langfuse；后端通过 OTLP/HTTP 写入 `LANGFUSE_OTEL_ENDPOINT`，不再保留 Phoenix。

Gemini 示例：

```bash
COMPOSE_GENERATION_API_KEY=<your-gemini-key>
COMPOSE_GENERATION_API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
COMPOSE_GENERATION_MODEL_ALIAS=gemini-flash-lite-latest

COMPOSE_EMBEDDING_PROVIDER=gemini
COMPOSE_EMBEDDING_API_KEY=<your-embedding-key>
COMPOSE_EMBEDDING_API_BASE_URL=https://generativelanguage.googleapis.com
COMPOSE_EMBEDDING_MODEL_ALIAS=gemini-embedding-2-preview

COMPOSE_DOCUMENT_PARSER_API_KEY=<your-parser-key>
COMPOSE_DOCUMENT_PARSER_API_BASE_URL=https://generativelanguage.googleapis.com
COMPOSE_DOCUMENT_PARSER_MODEL=gemini-flash-lite-latest
```

## 公开 RAG 评测

公开数据集用于通用 RAG 基线，规范/标准语料压测用于业务场景基线。可先抽样公开数据生成 `data/raw/public_eval/...` 和 `data/eval/*.jsonl`：

```bash
python scripts/prepare_public_rag_eval.py --dataset ragbench-delucionqa --limit 20
python scripts/prepare_public_rag_eval.py --dataset hotpotqa --limit 20
```

然后在 Docker 栈内摄入对应 raw 目录，并复用现有评测：

```bash
docker compose -f docker/docker-compose.yml exec app python scripts/run_eval.py --dataset data/eval/ragbench-delucionqa.jsonl
```

Qwen 示例：

```bash
COMPOSE_GENERATION_API_KEY=<your-dashscope-key>
COMPOSE_GENERATION_API_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
COMPOSE_GENERATION_MODEL_ALIAS=qwen-plus

COMPOSE_EMBEDDING_PROVIDER=dashscope
COMPOSE_EMBEDDING_API_KEY=<your-dashscope-key>
COMPOSE_EMBEDDING_API_BASE_URL=https://dashscope.aliyuncs.com
COMPOSE_EMBEDDING_MODEL_ALIAS=multimodal-embedding-v1

COMPOSE_DOCUMENT_PARSER_PROVIDER=qwen
COMPOSE_DOCUMENT_PARSER_API_KEY=<your-dashscope-key>
COMPOSE_DOCUMENT_PARSER_API_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
COMPOSE_DOCUMENT_PARSER_MODEL=qwen-vl-plus
```

Qwen vLLM 自托管示例：

```bash
COMPOSE_GENERATION_API_KEY=EMPTY
COMPOSE_GENERATION_API_BASE_URL=http://vllm:8000/v1
COMPOSE_GENERATION_MODEL_ALIAS=Qwen/Qwen2.5-VL-7B-Instruct

COMPOSE_EMBEDDING_PROVIDER=vllm
COMPOSE_EMBEDDING_API_KEY=EMPTY
COMPOSE_EMBEDDING_API_BASE_URL=http://vllm-embedding:8000/v1
COMPOSE_EMBEDDING_MODEL_ALIAS=Qwen/Qwen3-VL-Embedding-8B

COMPOSE_DOCUMENT_PARSER_PROVIDER=qwen
COMPOSE_DOCUMENT_PARSER_API_KEY=EMPTY
COMPOSE_DOCUMENT_PARSER_API_BASE_URL=http://vllm:8000/v1
COMPOSE_DOCUMENT_PARSER_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
```

可选 Qwen rerank：

```bash
COMPOSE_RERANK_MODEL_ALIAS=qwen3-rerank
COMPOSE_DISABLE_RERANK=false
COMPOSE_RERANK_API_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-api/v1
COMPOSE_RERANK_API_KEY=<your-dashscope-key>
COMPOSE_RERANK_API_PATH=/reranks
```

如果没有有效 provider key，系统不会切到 mock；`/health/detail` 会显示 degraded，并在具体能力下返回上游错误。

PDF 和图片上传需要 `DOCUMENT_PARSER_API_KEY`。未配置时后端启动日志会输出 `Startup readiness` 警告，`/health/detail` 会返回明确缺失项，上传接口也会直接拒绝；这不是回退或 mock，而是要求补齐真实 Gemini、Qwen DashScope 或 Qwen vLLM document parser 配置。

## 知识库与数据来源

知识库由后端 `/v1/rag/knowledge-bases` 生成，来源包括：

- 本地知识库 catalog，默认包含 `default`
- `RAG_SUPPORTED_KB_IDS` 启动时预置的知识库
- 前端或 API 创建的知识库

前端不会写死知识库。注入来源由 `/v1/rag/ingest/sources` 返回，默认来自 Docker 内的 `/workspace/data/raw` 白名单目录。

Nano RAG 不实现账号系统或 workspace/组织/项目管理；这些属于上层业务系统。未来外部账号系统可以通过网关或业务后端控制可访问的 `kb_id` 集合。

上传接口使用稳定的逻辑来源路径，例如 `uploads/default/policy.pdf`。Docker 会把原始上传文件持久化到 `app-upload-data` 卷里的 `/workspace/data/uploads`，用于多模态证据读取、后续重嵌入和可追溯 source path；临时批处理目录会在注入请求结束后清理。

## 常用验证

所有命令都通过 Docker 暴露的服务验证真实后端：

```bash
curl -sS http://127.0.0.1:3000/health/detail
curl -sS http://127.0.0.1:3000/v1/rag/knowledge-bases
curl -sS http://127.0.0.1:3000/v1/rag/ingest/sources
curl -sS http://127.0.0.1:3000/debug/storage
docker logs --tail 120 nano-rag-app
docker logs --tail 120 nano-rag-frontend
```

业务接口通过前端统一入口访问时由 nginx 注入业务 key。仅在容器网络内直连后端时才需要业务 API key：

```bash
docker compose -f docker/docker-compose.yml exec app \
  python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())"
curl -sS http://127.0.0.1:3000/health/detail \
  -H 'X-API-Key: nano-rag-local'
```

## API 示例

路径注入：

```bash
curl -sS -X POST http://127.0.0.1:3000/v1/rag/ingest \
  -H 'Content-Type: application/json' \
  -d '{"path":"/workspace/data/raw/employee_handbook.md","kb_id":"default"}'
```

问答：

```bash
curl -sS -X POST http://127.0.0.1:3000/v1/rag/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"请输入你的真实业务问题","kb_id":"default","session_id":"manual-session","top_k":4}'
```

仅检索上下文，不调用生成模型：

```bash
curl -sS -X POST http://127.0.0.1:3000/v1/rag/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"请输入你的真实检索问题","kb_id":"default","session_id":"manual-session","top_k":10}'
```

检索调试：

```bash
curl -sS -X POST http://127.0.0.1:3000/retrieve/debug \
  -H 'Content-Type: application/json' \
  -d '{"query":"请输入你的真实检索问题","kb_id":"default","top_k":10}'
```

## 关键配置

- [configs/settings.yaml](./configs/settings.yaml)：chunk、retrieval、hybrid search、timeout
- [configs/models.yaml](./configs/models.yaml)：各能力的 base_url、api_key 和模型 alias
- [configs/prompts.yaml](./configs/prompts.yaml)：生成提示词

注意：

- `MODEL_GATEWAY_MODE=mock` 不再是受支持的运行模式。
- `VECTORSTORE_BACKEND` 默认是 `milvus`；`memory` 仅保留给单元测试和显式实验。
- 默认 Docker 不启动模型网关中间层；生成、embedding、文档解析分别按显式 provider 配置直连。
- 文档解析不会回退到本地 PDF 解析器；PDF/图片解析需要启用并配置 document parser。
- embedding 不会把多模态输入降级成文本；embedding client 必须支持 `embed_items`。
- freshness 过滤不会追加旧版本内容作为兜底上下文；需要通过显式 `source_key` 管理版本组。

## 测试

代码检查可以直接运行测试命令；服务启动和联调只使用 Docker：

```bash
python -m pytest app/tests
npm --prefix frontend run lint
npm --prefix frontend run build
docker compose -f docker/docker-compose.yml up -d --build
```

当前真实运行状态以 Docker 为准。没有有效 provider key 时，构建和服务健康可以通过，但注入/问答会在模型调用阶段返回真实上游错误。
