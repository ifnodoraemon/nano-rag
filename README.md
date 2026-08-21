# Nano RAG

Nano RAG 是一个真实数据优先的企业 RAG 工作台。后端负责文档注入、解析、分块、**文档级 BM25 发现**、agentic 检索与深读、生成、追踪、评测和诊断；前端只展示后端已经支持的能力，不保存业务密钥，不内置 mock 数据，也不通过写死选项伪造状态。

![Nano RAG 运营台](./nanorag.png)

## 快速开始

```bash
docker compose -f docker/docker-compose.yml up -d --build
```

访问地址：

- 前端/API 统一入口：`http://127.0.0.1:3001`

默认只暴露前端一个 host 端口；后端、PostgreSQL、Redis 和 Celery worker 都只在 Docker 网络内通信。

如果只做本地 UI/接口验证，不需要 PostgreSQL 和 Celery worker，可使用 lite 覆盖文件，只启动 `app` 和 `frontend` 两个容器：

```bash
docker compose -f docker/docker-compose.lite.yml up -d --build
```

lite 模式使用 artifact 图扩展（无数据库）和 background ingest；正式规范压测仍建议使用默认生产栈（postgres 图存储）。

浏览器端不需要配置 API key。前端 nginx 会为代理到后端的请求注入业务 key；本地 Docker 默认使用 `RAG_PROXY_API_KEY=nano-rag-local`，需要和后端 `RAG_API_KEYS` 中的一个值保持一致。

启动后优先检查真实运行状态：

```bash
curl -sS http://127.0.0.1:3001/health/detail
curl -sS http://127.0.0.1:3001/v1/rag/knowledge-bases
curl -sS http://127.0.0.1:3001/v1/rag/ingest/sources
```

## 项目理念

Nano RAG 的核心原则是：**真实输入、真实索引、真实模型、真实错误**。

- **不 mock**：运行时不使用 mock 网关，不把假数据展示成可用能力。
- **不回退**：generation、document parser 都必须显式配置；缺少配置或上游失败时直接暴露错误。
- **无稠密向量回退**：检索引擎是文档级 BM25 发现 + agentic 深读，不存在稠密/混合向量路径作为静默兜底。
- **后端是事实来源**：知识库、可注入文件、文档列表、追踪、评测数据集、报告和诊断对象都来自后端接口。
- **前端不配置密钥**：浏览器通过上层业务后端、网关或 Docker 代理与后端连接；本地 Docker 代理会注入默认业务 API key。
- **知识库优先**：所有业务操作都挂在 `kb_id` 表示的知识库下，前端必须先选择后端返回的知识库。
- **证据优先**：回答必须带引用、上下文和 trace，问题排查必须能回到原始文档、分块和模型请求链路。

这意味着系统更愿意失败得明确，也不做静默降级。错误的 API key、不可用的解析器、不可达的 PostgreSQL、空的 provider key，都应该被看见和修复。

## 当前能力

- 业务 API：`/v1/rag/chat`、`/v1/rag/chat/stream`、`/v1/rag/retrieve`、`/v1/rag/ingest`、`/v1/rag/ingest/upload`、`/v1/rag/ingest/jobs/{job_id}`、`/v1/rag/documents`、`/v1/rag/knowledge-bases`、`/v1/rag/ingest/sources`、`/v1/rag/feedback`、`/v1/rag/traces/{trace_id}`
- 文档结构 API：`/v1/rag/documents/{doc_id}/tree`、`/v1/rag/nodes/{node_id}`、`/v1/rag/tables/{table_id}`、`/v1/rag/graph/neighborhood`
- 运维 API：`/health/detail`、`/debug/storage`、`/debug/parsed/{doc_id}`、`/retrieve/debug`、`/traces`、`/replay/{trace_id}`
- 评测和诊断：`/eval/datasets`、`/eval/reports`、`/eval/run`、`/benchmark/reports`、`/diagnose/*`
- 发现层：文档级 BM25 索引（编译后的 wiki 页），是 agentic 检索的第一跳；无向量库、无 embedding 模型
- 图存储：原生 PostgreSQL 物化文档结构（节点/实体/关系），支撑可选的图扩展；本地脚本可退回 artifact 存储
- 摄入解析：内置 parser registry，支持 Markdown、TXT、HTML、CSV/TSV、JSON/JSONL、YAML、XML、DOCX、XLSX、PPTX、数字版 PDF、本地媒体文件；旧版 DOC/XLS/PPT、扫描 PDF 和图片重文档通过已配置的多模态 document parser 解析
- 文档结构：保留章节、条款、定义、表格 summary、表格 row-level chunk、来源 hash、文件版本元数据、parser 名称、chunk_strategy 和索引 schema 版本
- 图片索引：图片文件默认生成视觉媒体 chunk；如果配置了多模态 document parser 且能抽出文字，会同时生成文本/结构 chunk
- agentic 检索：wiki BM25 发现 → 确定性版本过滤（按 `source_key` 取最新）→ LLM 结构化读计划（json_schema）→ 深读选中的 parsed artifact → 上下文构建；每个阶段进 trace
- 证据治理：生成结果带 structured supporting claims，并在后处理里标记 claim 是否有引用支撑、缺失数字和缺失术语
- 模型路径：生成、文档解析、可选 rerank 都直连显式配置的真实 provider；默认示例只覆盖 Gemini 和 Qwen；**不配置 embedding 端点**
- 前端：React + Vite 源码位于 `frontend/` 子模块，Docker 构建为 nginx 静态站点并代理后端

## 目录

```text
nano-rag/
├─ app/              # FastAPI 后端（agentic 发现 + 检索 + 生成）
├─ configs/          # settings/models/prompts
├─ data/             # raw/eval/wiki 样例数据
├─ docker/           # Dockerfile、Compose、nginx 配置
├─ frontend/         # React + Vite 前端子模块
└─ scripts/          # eval/benchmark/smoke 等脚本
```

## Docker 配置

Docker 默认值：

```bash
MODEL_GATEWAY_MODE=live
GENERATION_API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
RAG_API_KEYS=nano-rag-local
RAG_GRAPH_BACKEND=postgres
PG_URI=postgresql://nanorag:nano-rag@postgres:5432/nanorag
RAG_WIKI_ENABLED=true
RAG_DIAGNOSIS_ENABLED=true
RAG_EVAL_ENABLED=true
DOCUMENT_PARSER_ENABLED=true
```

后端启动时会检查 generation、document parser 等关键配置，并连接 PostgreSQL 图存储。缺少 `DOCUMENT_PARSER_API_KEY` 这类真实 provider 配置时，容器日志会输出 `Startup readiness` 警告，`/health/detail` 会继续显示对应能力不可用；前端只消费这些后端状态，不负责提示 Docker 配置方式。

## Provider 配置

真实模型调用需要配置 provider key。默认工程不再启动 Bifrost 或 LiteLLM，也不读取它们的密钥文件。只保留两套推荐配置：Gemini 或 Qwen。

模型 provider 抽象按能力拆分：

- `generation`：OpenAI-compatible chat completions，支持 Gemini、Qwen DashScope、Qwen vLLM。
- `document_parser`：`gemini` 使用 Gemini Files API；`qwen` 使用 OpenAI-compatible chat completions，可指向 DashScope 或 vLLM。
- `rerank`：默认关闭；需要时显式配置 Qwen rerank endpoint 和 path。
- `trace`：默认写入本地 `TraceStore`，可选通过 `LANGFUSE_OTEL_ENDPOINT` 对接外部 Langfuse；默认 Docker 栈不再内置 Langfuse 容器。

Gemini 示例：

```bash
COMPOSE_GENERATION_API_KEY=<your-gemini-key>
COMPOSE_GENERATION_API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
COMPOSE_GENERATION_MODEL_ALIAS=gemini-flash-latest

COMPOSE_DOCUMENT_PARSER_API_KEY=<your-parser-key>
COMPOSE_DOCUMENT_PARSER_API_BASE_URL=https://generativelanguage.googleapis.com
COMPOSE_DOCUMENT_PARSER_MODEL=gemini-flash-latest
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

上传接口使用稳定的逻辑来源路径，例如 `uploads/default/policy.pdf`。Docker 会把原始上传文件持久化到 `app-upload-data` 卷里的 `/workspace/data/uploads`，用于证据读取和可追溯 source path；临时批处理目录会在注入请求结束后清理。

## 常用验证

所有命令都通过 Docker 暴露的服务验证真实后端：

```bash
curl -sS http://127.0.0.1:3001/health/detail
curl -sS http://127.0.0.1:3001/v1/rag/knowledge-bases
curl -sS http://127.0.0.1:3001/v1/rag/ingest/sources
curl -sS http://127.0.0.1:3001/debug/storage
docker logs --tail 120 nanorag-app
docker logs --tail 120 nanorag-frontend
```

业务接口通过前端统一入口访问时由 nginx 注入业务 key。仅在容器网络内直连后端时才需要业务 API key：

```bash
docker compose -f docker/docker-compose.yml exec app \
  python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())"
curl -sS http://127.0.0.1:3001/health/detail \
  -H 'X-API-Key: nano-rag-local'
```

## API 示例

路径注入：

```bash
curl -sS -X POST http://127.0.0.1:3001/v1/rag/ingest \
  -H 'X-RAG-Admin-Key: <your-admin-key>' \
  -H 'Content-Type: application/json' \
  -d '{"path":"/workspace/data/raw/employee_handbook.md","kb_id":"default"}'
```

问答：

```bash
curl -sS -X POST http://127.0.0.1:3001/v1/rag/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"请输入你的真实业务问题","kb_id":"default","session_id":"manual-session","top_k":4}'
```

仅检索上下文，不调用生成模型：

```bash
curl -sS -X POST http://127.0.0.1:3001/v1/rag/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"请输入你的真实检索问题","kb_id":"default","session_id":"manual-session","top_k":10}'
```

检索调试：

```bash
curl -sS -X POST http://127.0.0.1:3001/retrieve/debug \
  -H 'X-RAG-Admin-Key: <your-admin-key>' \
  -H 'Content-Type: application/json' \
  -d '{"query":"请输入你的真实检索问题","kb_id":"default","top_k":10}'
```

## 生产部署前置项

本仓库默认配置面向本地验证。上线前至少完成以下调整：

- 设置 `RAG_ENV=production`，并替换 `RAG_API_KEYS`、`RAG_ADMIN_API_KEYS`、`PG_PASSWORD` 等所有默认密钥；生产环境会拒绝 `nano-rag-local` 这类默认 API key。
- 业务接口使用 `RAG_API_KEYS` 或可信上游网关；管理、调试、评测、诊断和 replay 接口使用独立的 `RAG_ADMIN_API_KEYS`。
- 如果由业务网关做登录和租户控制，设置 `RAG_TRUSTED_PROXY_SECRET`，并让网关注入 `X-RAG-Proxy-Secret`、`X-RAG-Principal-Id`、`X-RAG-Org-Id`、`X-RAG-Allowed-KB-Ids`。
- 设置 `RAG_RATE_LIMIT_REQUESTS_PER_MINUTE`，并在外层网关或 WAF 做全局限流；仓库内的限流是单进程保护，不替代分布式限流。
- 为 PostgreSQL 图存储和 `app-parsed-data`、`app-report-data`、`app-upload-data`、`app-wiki-data` 卷建立备份、恢复和迁移流程。
- 默认不再保存完整 prompt 到 trace；只有设置 `RAG_TRACE_STORE_PROMPTS=true` 时才会写入 `prompt_messages`。
- 模型 provider 默认对 408/409/425/429/5xx 和超时做退避重试；可通过 `MODEL_PROVIDER_MAX_RETRIES`、`MODEL_PROVIDER_RETRY_BASE_SECONDS`、`MODEL_PROVIDER_RETRY_MAX_SECONDS` 调整。

生产常见文档处理：

- 直接支持：Markdown、TXT、HTML、CSV、DOCX、XLSX、图片、音频、视频、PDF。
- CSV/DOCX/XLSX 使用本地结构化抽取，保留基础段落、表格和单元格值；复杂版式、扫描件、图片型 PDF、图表型材料仍应配置多模态 document parser。
- 上传大小由 `MAX_UPLOAD_BYTES` 控制，单批文件数由 `MAX_FILES_PER_BATCH` 控制；生产应在网关层同步设置 body size、杀毒/内容安全扫描和文件类型白名单。
- 对标准、法规、合同、制度类文档，建议在文件名或正文中保留版本号、生效日期、章节号；发现层会使用 `doc_type`、`effective_date`、`version`、`source_key` 等元数据做确定性版本过滤。

评测门禁示例：

```bash
python scripts/run_eval.py \
  --dataset data/eval/employee_handbook_eval.jsonl \
  --min-context-recall 0.85 \
  --max-conflicting-hit-rate 0.05
```

RAG 本体回归样本位于 `data/eval/rag_quality_regression.jsonl`，覆盖表格行查值、条款定位、术语定义、版本冲突和无答案边界。生产接入新语料后，应把业务高风险问题追加到这个集合并设为 CI 门禁。

## 关键配置

- [configs/settings.yaml](./configs/settings.yaml)：agentic 检索、discovery、prompt、timeout
- [configs/models.yaml](./configs/models.yaml)：各能力的 base_url、api_key 和模型 alias
- [configs/prompts.yaml](./configs/prompts.yaml)：生成提示词

注意：

- `MODEL_GATEWAY_MODE=mock` 不再是受支持的运行模式。
- `RAG_GRAPH_BACKEND` 默认生产栈是 `postgres`；本地脚本默认 `artifact`（无需数据库）。
- 检索引擎是文档级 BM25 发现 + agentic 深读；`RAG_WIKI_ENABLED` 必须为 `true`，不存在稠密向量回退。
- 默认 Docker 不启动模型网关中间层；生成、文档解析分别按显式 provider 配置直连，**没有 embedding 端点**。
- 文档解析不会回退到本地 PDF 解析器；PDF/图片解析需要启用并配置 document parser。
- 附件索引默认开启：`RAG_DOCUMENT_ATTACHMENT_INDEX_ENABLED=true`。PDF page attachment 页数上限由 `RAG_PDF_ATTACHMENT_MAX_PAGES` 控制；PDF rendered image 需要运行环境存在 `pdftoppm`，并可用 `RAG_RENDERED_PAGE_IMAGE_INDEX_ENABLED`、`RAG_RENDERED_PAGE_IMAGE_DPI` 控制；OOXML 内嵌图片由 `RAG_EMBEDDED_IMAGE_INDEX_ENABLED`、`RAG_EMBEDDED_IMAGE_MAX_COUNT` 控制。

## 测试

代码检查可以直接运行测试命令；服务启动和联调只使用 Docker：

```bash
python -m pytest app/tests
npm --prefix frontend run lint
npm --prefix frontend run build
docker compose -f docker/docker-compose.yml up -d --build
```

当前真实运行状态以 Docker 为准。没有有效 provider key 时，构建和服务健康可以通过，但注入/问答会在模型调用阶段返回真实上游错误。
