# Nano RAG

根据 `rag_project_plan_v2.md` 初始化的企业级 RAG 项目骨架，当前定位为
**以 LLM Wiki 为知识底座的轻量企业 RAG 系统**。

## 项目理念

Nano RAG 是最小可用的问答运行时，LLM Wiki 是长期知识记忆和治理层。
本项目不追求把所有能力都默认打开，而是把能力分成三层：

- **Nano Core**：`ingest -> retrieve -> answer -> cite -> trace`，默认路径要轻、清楚、可本地验证
- **LLM Wiki Layer**：把来源文档、主题、事实、规则、冲突、版本和适用条件沉淀成可追溯知识资产
- **Workbench**：用 eval、benchmark、diagnosis、feedback 找坏例，并把修复反哺到文档和 wiki

设计取舍：

- 回答必须证据优先：每次回答都要能追到来源、上下文和 trace
- 本地体验要低摩擦：可以显式关闭业务鉴权，用 mock 网关先跑通核心链路
- 生产默认要严肃：共享或生产环境默认要求业务 API key，不把未配置 key 当作开放访问
- 可选能力不压主流程：Milvus、Phoenix、wiki、eval、diagnosis 都是增强层，不阻塞 nano core

当前版本提供：

- `FastAPI` 服务，核心公开入口为 `/health`
- 调试/评测 API：`/retrieve/debug`、`/traces`、`/eval/*`、`/benchmark/*`、`/diagnose/*`
- 业务接入 API：`/v1/rag/chat`、`/v1/rag/ingest`、`/v1/rag/ingest/upload`、`/v1/rag/feedback`、`/v1/rag/traces/{trace_id}`、`/v1/rag/benchmark/run`
- `ingestion / retrieval / generation` 主链路模块拆分
- 统一 `model_client`，所有模型调用都走 OpenAI-compatible Gateway
- 默认本地走 `memory` 向量仓储，可切换到 `Milvus`
- `Phoenix`、`Milvus`、`benchmark/eval/diagnosis` 都保留，但不再要求本地最小启动必须依赖
- `Docker Compose` 骨架，包含 `app / milvus / phoenix / frontend (nginx)`
- 基础测试、trace 存储、配置模板、离线评测和 benchmark 脚手架

## 目录

```text
nano-rag/
├─ docker/
├─ configs/
├─ data/
├─ app/
└─ frontend/        # React + Vite 前端
```

## 快速开始

### 1. 本地运行

```bash
cd nano-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export MODEL_GATEWAY_MODE=mock
export RAG_AUTH_DISABLED=true
export RAG_INGEST_ALLOWED_DIRS=$PWD/data/raw
uvicorn app.main:app --reload --app-dir .
```

说明：

- 本地最小启动默认推荐 `MODEL_GATEWAY_MODE=mock`，并显式设置 `RAG_AUTH_DISABLED=true` 先把 ingest / retrieve / trace 跑通
- 如果要直连真实模型，把 `MODEL_GATEWAY_MODE` 改成 `live`，并配置 `MODEL_GATEWAY_BASE_URL / MODEL_GATEWAY_API_KEY`
- ingest 默认要求显式配置白名单目录；上面的 `RAG_INGEST_ALLOWED_DIRS=$PWD/data/raw` 是为了让 README 里的示例可直接运行
- 共享或生产环境不要设置 `RAG_AUTH_DISABLED=true`，请改用 `RAG_API_KEYS=<key1,key2>`
- 当前默认只启用 nano core。下面这些能力现在需要显式打开：

```bash
export RAG_WIKI_ENABLED=true
export RAG_DIAGNOSIS_ENABLED=true
export RAG_EVAL_ENABLED=true
```

- `RAG_HYBRID_SEARCH_ENABLED`、`RAG_SEMANTIC_CHUNKER_ENABLED`、`RAG_QUERY_REWRITE_ENABLED` / `RAG_MULTI_QUERY_ENABLED` / `RAG_HYDE_ENABLED` 也都属于可选增强，不再默认启动

### 2. 启动 React 前端

前端采用 React + Vite，与后端分离：

```bash
cd nano-rag/frontend
npm install
npm run dev
```

前端启动后访问 http://127.0.0.1:5173 ，会自动代理到后端 API。
如果后端不在本机 `8000`，可覆盖：

```bash
cd nano-rag/frontend
VITE_DEV_API_TARGET=http://your-backend-host:8000 npm run dev
```

前端主页现在按 NanoRAG 的业务验证流程组织：

- 工作区配置：`kb_id / tenant_id / session_id / business api key`
- 导入知识：调用 `/v1/rag/ingest`
- 问答验证：调用 `/v1/rag/chat`
- 反馈回流：调用 `/v1/rag/feedback`
- 高级区：`retrieve/debug`、`traces`、`eval`、`benchmark`
- 业务 API 与高级调试 API 默认需要鉴权；配置 `RAG_API_KEYS` 后，前端工作区里填写其中一枚 key
- 仅本地开发可设置 `RAG_AUTH_DISABLED=true` 跳过业务鉴权，此时前端 API Key 可以留空
- 前端不会把业务 API Key 持久化到本地存储，刷新页面后需要重新输入

### 3. Docker Compose

```bash
cd nano-rag
docker compose -f docker/docker-compose.yml up -d --build
```

Docker 模式下：

- React 前端: `http://127.0.0.1:3000`
- FastAPI 后端: `http://127.0.0.1:8000`
- Phoenix: `http://127.0.0.1:6006`
- Milvus: `http://127.0.0.1:19530`

Compose 默认启用真实向量库和 Phoenix：

```bash
VECTORSTORE_BACKEND=milvus
MILVUS_URI=http://milvus:19530
```

鉴权策略与本地运行一致：共享或生产环境在 `.env` 里配置 `RAG_API_KEYS`；
只做本机开发验证时，可以在 `.env` 里显式设置 `RAG_AUTH_DISABLED=true`。

### 4. 直连外部模型接口

外部 provider 提供 OpenAI-compatible 接口，直接配置根目录 `.env`：

```bash
export MODEL_GATEWAY_MODE=live
export MODEL_GATEWAY_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
export MODEL_GATEWAY_API_KEY=<your-gemini-api-key>
export DISABLE_RERANK=1
```

推荐同时把 [models.yaml](/home/ifnodoraemon/myagent/nano-rag/configs/models.yaml) 改成：

```yaml
embedding:
  default_alias: gemini-embedding-001
  dimension: 3072

rerank:
  default_alias: disabled

generation:
  default_alias: gemini-2.5-flash
```

说明：

- Gemini API 当前可直接覆盖 `/chat/completions` 和 `/embeddings`
- 当前项目的 `/rerank` 阶段需要单独的 rerank provider；若走 Gemini API，请关闭 rerank
- 业务 API 中的 `kb_id` 已预留，但当前部署仅支持 `default`

### 5. 按能力拆分模型网关

如果 generation / embedding / rerank 需要走不同 provider，现在可以分别配置不同的 `base_url + key`。
未单独配置时，会自动回退到全局 `MODEL_GATEWAY_BASE_URL / MODEL_GATEWAY_API_KEY`。

```bash
export MODEL_GATEWAY_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
export MODEL_GATEWAY_API_KEY=<gemini-api-key>

export EMBEDDING_API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
export EMBEDDING_API_KEY=<gemini-api-key>

export RERANK_API_BASE_URL=https://your-rerank-provider.example.com/v1
export RERANK_API_KEY=<your-rerank-api-key>
```

例如：

- `generation` 走 Gemini API
- `embedding` 走 Gemini API
- `rerank` 走独立 rerank provider

对应 [models.yaml](/home/ifnodoraemon/myagent/nano-rag/configs/models.yaml) 里的 alias 也可以分开维护。

### 6. 可选：通过 Bifrost 统一多家 provider

只有当你需要把多家 provider 统一成一个网关地址、做负载均衡或故障转移时，才需要启用 Bifrost。

Bifrost 支持 OpenAI Chat Completions、OpenAI Responses API、Anthropic Messages 三种格式，可透明路由到 15+ provider。

```bash
cp docker/bifrost/.env.example docker/bifrost/.env
# 编辑 docker/bifrost/.env 填入 provider API key
# 按需编辑 docker/bifrost/config.json 配置 provider 和模型列表
docker compose -f docker/docker-compose.yml --profile bifrost up -d bifrost
```

然后把 `.env` 里的 `MODEL_GATEWAY_BASE_URL` 指到 `http://bifrost:8080/v1`。

Bifrost 还自带 Web UI（`http://127.0.0.1:8080`），可可视化管理 provider 配置。

## 示例

### health

```bash
curl http://127.0.0.1:8000/health
```

详细健康信息走受保护接口：

```bash
curl http://127.0.0.1:8000/health/detail \
  -H 'X-API-Key: <your-business-api-key>'
```

如果本地设置了 `RAG_AUTH_DISABLED=true`，可省略 `X-API-Key`。

### business ingest

```bash
curl -X POST http://127.0.0.1:8000/v1/rag/ingest \
  -H 'Content-Type: application/json' \
  -d '{"path":"./data/raw","kb_id":"default","tenant_id":"demo-tenant"}'
```

### business chat

```bash
curl -X POST http://127.0.0.1:8000/v1/rag/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"差旅报销多久内提交？","kb_id":"default","tenant_id":"demo-tenant","session_id":"session-001"}'
```

共享或生产环境需要在受保护 API 上加：

```bash
-H 'X-API-Key: <your-business-api-key>'
```

### business feedback

```bash
curl -X POST http://127.0.0.1:8000/v1/rag/feedback \
  -H 'Content-Type: application/json' \
  -d '{"trace_id":"<trace_id>","rating":"up","kb_id":"default","tenant_id":"demo-tenant","comment":"answer is helpful"}'
```

### debug retrieval

```bash
curl -X POST http://127.0.0.1:8000/retrieve/debug \
  -H 'Content-Type: application/json' \
  -d '{"query":"差旅报销多久内提交？","top_k":10}'
```

### run eval by API

先确认已经设置 `RAG_EVAL_ENABLED=true`。

```bash
curl -X POST http://127.0.0.1:8000/eval/run \
  -H 'Content-Type: application/json' \
  -d '{"dataset_path":"data/eval/employee_handbook_eval.jsonl","output_path":"data/samples/eval_report_api.json"}'
```

### traces

```bash
curl http://127.0.0.1:8000/traces
curl http://127.0.0.1:8000/traces/<trace_id>
```

### storage debug

```bash
curl http://127.0.0.1:8000/debug/storage
curl http://127.0.0.1:8000/debug/parsed/<doc_id>
```

### offline eval

```bash
python3 scripts/run_eval.py \
  --dataset data/eval/employee_handbook_eval.jsonl \
  --output data/samples/eval_report.json
```

### benchmark

先确认已经同时设置 `RAG_EVAL_ENABLED=true` 和 `RAG_DIAGNOSIS_ENABLED=true`。

```bash
curl -X POST http://127.0.0.1:8000/v1/rag/benchmark/run \
  -H 'Content-Type: application/json' \
  -d '{"dataset_path":"data/eval/employee_handbook_eval.jsonl"}'
```

```bash
python3 scripts/run_benchmark.py \
  --dataset data/eval/employee_handbook_eval.jsonl \
  --output data/samples/benchmark_report.json
```

### benchmark reports

```bash
curl http://127.0.0.1:8000/benchmark/reports
curl 'http://127.0.0.1:8000/benchmark/reports/detail?path=data/reports/eval/benchmarks/<report>.json'
```

离线评测数据集至少应包含：

- `query`
- `reference_answer`
- `reference_contexts`

如果未提供 `answer` 或 `retrieved_contexts`，当前 `/eval/run` API 和 `scripts/run_eval.py` 会自动走现有 RAG 链路生成候选答案与召回上下文，再计算指标。

## 配置说明

- [settings.yaml](/home/ifnodoraemon/myagent/nano-rag/configs/settings.yaml): chunk、retrieval、timeout 等运行参数
- [models.yaml](/home/ifnodoraemon/myagent/nano-rag/configs/models.yaml): gateway 地址和模型 alias
- [prompts.yaml](/home/ifnodoraemon/myagent/nano-rag/configs/prompts.yaml): 生成提示词模板

## 现阶段说明

- 本地默认向量仓储使用 `memory`
- Docker Compose 默认把 `VECTORSTORE_BACKEND` 设成 `milvus`
- PDF 优先走 `Docling`，未安装时回退 `pypdf`
- `Phoenix` 现在是可选项；未配置 endpoint 时不会影响主流程健康状态
- `RAGAS` 相关脚本已接入离线评测入口，当前默认使用确定性聚合指标，后续可替换成真实 RAGAS 流程
- 当前已补充 benchmark 服务和脚本，用于聚合离线质量、延迟和坏例诊断统计
- 前端主流程默认走业务 API；共享或生产环境需要在前端工作区中填写 `RAG_API_KEYS` 中的一枚 key
- 默认建议本地使用 `MODEL_GATEWAY_MODE=mock`；接真实 provider 时再切到 `live`
- 直连外部 OpenAI-compatible provider
- 如果直连 Gemini API，请设置 `MODEL_GATEWAY_API_KEY`，并通过 `DISABLE_RERANK=1` 跳过 rerank 阶段
- 仅当需要网关聚合能力时，再启用 `docker compose --profile bifrost`

## 测试

```bash
cd nano-rag
pytest app/tests
```

前端和交付前验证要求：

```bash
cd nano-rag/frontend
npm run lint
npm run build
```

Docker 运行也属于交付要求。涉及后端、前端、配置、依赖或部署行为的改动，提交前需要至少完成一次 Docker Compose 运行验证，或在交付说明里明确说明为什么本轮无法运行：

```bash
cd nano-rag
docker compose -f docker/docker-compose.yml up -d --build
curl http://127.0.0.1:8000/health
docker compose -f docker/docker-compose.yml down
```
