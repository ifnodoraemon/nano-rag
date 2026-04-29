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
- `Phoenix`、`Milvus`、`Bifrost`、`benchmark/eval/diagnosis` 都保留；本地最小启动可用 mock，Docker 默认走生产形态
- Docker 默认启动 Bifrost，模型 alias 通过网关路由；rerank 在配置 provider key 后启用
- Milvus 新 collection 默认包含 analyzer text 字段、BM25 sparse 字段和原生 hybrid search 路径
- `Docker Compose` 骨架，包含 `app / milvus / phoenix / frontend (nginx)`；前端镜像在构建时从上游 [`ifnodoraemon/nano-rag-ui`](https://github.com/ifnodoraemon/nano-rag-ui) 拉取并打包，本仓库不再保存前端源码
- 基础测试、trace 存储、配置模板、离线评测和 benchmark 脚手架

## 目录

```text
nano-rag/
├─ docker/         # Compose 与 Dockerfile（含前端构建时 clone 上游 UI 仓库）
├─ configs/
├─ data/
└─ app/            # FastAPI 后端
```

前端 React + Vite 源码在独立仓库 [`ifnodoraemon/nano-rag-ui`](https://github.com/ifnodoraemon/nano-rag-ui) 维护。

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
- ingest 默认要求显式配置白名单目录；仓库内置 `data/raw/employee_handbook.md`，上面的 `RAG_INGEST_ALLOWED_DIRS=$PWD/data/raw` 可直接跑通 README 示例
- 共享或生产环境不要设置 `RAG_AUTH_DISABLED=true`，请改用 `RAG_API_KEYS=<key1,key2>`
- 本地 mock 模式用于验证 nano core；下面这些治理和诊断能力需要显式打开：

```bash
export RAG_WIKI_ENABLED=true
export RAG_DIAGNOSIS_ENABLED=true
export RAG_EVAL_ENABLED=true
```

- Docker 生产形态默认启用 Milvus 原生 hybrid search；本地可用 `RAG_HYBRID_SEARCH_ENABLED=false` 显式关闭
- `RAG_SEMANTIC_CHUNKER_ENABLED`、`RAG_QUERY_REWRITE_ENABLED` / `RAG_MULTI_QUERY_ENABLED` / `RAG_HYDE_ENABLED` 属于可选增强，不阻塞 nano core

### 2. 启动前端（独立仓库）

前端 UI 源码在 [`ifnodoraemon/nano-rag-ui`](https://github.com/ifnodoraemon/nano-rag-ui)。本地开发时单独 clone 并启动：

```bash
git clone https://github.com/ifnodoraemon/nano-rag-ui.git
cd nano-rag-ui
npm install
npm run dev
```

前端通过 Vite dev server 代理到后端（默认 `http://localhost:8000`），调用 `/v1/rag/**` 业务 API。
鉴权：`RAG_API_KEYS` 配置后在前端工作区填入其中一枚 key；本地可设置 `RAG_AUTH_DISABLED=true` 跳过。

### 3. Docker Compose

```bash
cd nano-rag
docker compose -f docker/docker-compose.yml up -d --build
```

前端镜像默认从 `ifnodoraemon/nano-rag-ui` 的 `main` 分支构建。要锁定版本或换源：

```bash
UI_REF=<commit-or-tag> UI_REPO_URL=https://github.com/<your-fork>/nano-rag-ui.git \
  docker compose -f docker/docker-compose.yml build frontend
```

Docker 模式下：

- React 前端: `http://127.0.0.1:3000`（构建时从 `nano-rag-ui` 上游 clone，可用 `UI_REF=<commit>` 锁定版本）
- FastAPI 后端: `http://127.0.0.1:8000`
- Phoenix: `http://127.0.0.1:6006`
- Milvus: `http://127.0.0.1:19530`

Compose 默认启用真实向量库、Phoenix 和 Bifrost：

```bash
VECTORSTORE_BACKEND=milvus
MILVUS_URI=http://milvus:19530
MODEL_GATEWAY_MODE=live
MODEL_GATEWAY_BASE_URL=http://bifrost:8080/v1
GENERATION_MODEL_ALIAS=gemini/gemini-3.1-pro-preview
EMBEDDING_MODEL_ALIAS=gemini-embedding-2-preview
EMBEDDING_API_BASE_URL=https://generativelanguage.googleapis.com
```

鉴权策略与本地运行一致：共享或生产环境在 `.env` 里配置 `RAG_API_KEYS`；
只做本机开发验证时，可以在 `.env` 里显式设置 `RAG_AUTH_DISABLED=true`。
真实模型调用需要在 `docker/bifrost/.env` 中配置 provider key，例如 `GEMINI_API_KEY`、`COHERE_API_KEY` 或 `OPENAI_API_KEY`。`app` 服务也会读取 `docker/bifrost/.env`，多模态 embedding 客户端通过其中的 `GEMINI_API_KEY` 直连 Gemini API（不经过 Bifrost）。
Docker Compose 的模型网关变量使用 `COMPOSE_*` 前缀覆盖，因此根目录 `.env` 直连 Gemini 时不会影响 Docker 默认走 Bifrost。
未配置 rerank provider key 时，Compose 默认把 `RERANK_MODEL_ALIAS` 置为 `disabled`；要启用 rerank，请设置 `COMPOSE_RERANK_MODEL_ALIAS=cohere/rerank-v3.5` 并在 `docker/bifrost/.env` 配置 `COHERE_API_KEY`。
PDF 解析仍走 Bifrost GenAI 的 multipart file upload + `generateContent`；图片不再走"图片→Markdown"路径，而是直接生成 image-modality chunk 并由多模态 embedding 索引到同一向量空间。

### 4. 直连外部模型接口

外部 provider 提供 OpenAI-compatible 接口，直接配置根目录 `.env`：

```bash
export MODEL_GATEWAY_MODE=live
export MODEL_GATEWAY_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
export MODEL_GATEWAY_API_KEY=<your-gemini-api-key>
export GENERATION_MODEL_ALIAS=gemini-3.1-pro-preview
# 多模态 embedding 直连 Gemini，不走 Bifrost / OpenAI-compat
export EMBEDDING_MODEL_ALIAS=gemini-embedding-2-preview
export EMBEDDING_API_BASE_URL=https://generativelanguage.googleapis.com
export EMBEDDING_API_KEY=<your-gemini-api-key>
export DISABLE_RERANK=1
```

推荐同时把 [models.yaml](/home/ifnodoraemon/myagent/nano-rag/configs/models.yaml) 改成：

```yaml
embedding:
  provider: gemini
  multimodal: true
  default_alias: gemini-embedding-2-preview
  dimension: 1536        # Matryoshka 截断；可选 768 / 1536 / 3072
  base_url: https://generativelanguage.googleapis.com

rerank:
  default_alias: disabled

generation:
  default_alias: gemini-3.1-pro-preview
```

说明：

- Embedding 走 **Gemini Embedding 2-preview**，原生跨模态（text / image / document / audio / video 共享一个向量空间）。客户端使用 `:embedContent` REST 路径，绕过 Bifrost 与 OpenAI `/embeddings`。
- Gemini API 当前 `/chat/completions` 走 Bifrost 或 `/v1beta/openai/chat/completions`；rerank 需要独立 provider，若走 Gemini API 请关闭。
- 升级提示：Gemini Embedding 1 与 2 的向量空间**不兼容**。从旧 collection 切到 1536 维 multimodal collection 必须 `docker compose down -v` 清空 milvus volume 后重新 ingest。
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

Bifrost 是 Docker Compose 默认模型网关；只有本地 mock 或直连单一 OpenAI-compatible provider 时才需要绕过它。

Bifrost 支持 OpenAI Chat Completions、OpenAI Responses API、Anthropic Messages 三种格式，可透明路由到 15+ provider。

```bash
cp docker/bifrost/.env.example docker/bifrost/.env
# 编辑 docker/bifrost/.env 填入 provider API key
# 按需编辑 docker/bifrost/config.json 配置 provider 和模型列表
docker compose -f docker/docker-compose.yml up -d bifrost
```

然后把 `.env` 里的 `MODEL_GATEWAY_BASE_URL` 保持为默认的 `http://bifrost:8080/v1`。

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

如需调用 RAGAS 库标准指标（faithfulness / answer_relevancy / context_precision），把 `use_ragas_lib` 设为 `true`，并确保评测 LLM 可通过模型网关访问：

```bash
curl -X POST http://127.0.0.1:8000/eval/run \
  -H 'Content-Type: application/json' \
  -d '{"dataset_path":"data/eval/employee_handbook_eval.jsonl","use_ragas_lib":true}'
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

真实 RAGAS 库指标：

```bash
python3 scripts/run_eval.py \
  --dataset data/eval/employee_handbook_eval.jsonl \
  --ragas-lib
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
- `RAGAS` 相关脚本已接入离线评测入口；默认使用确定性聚合指标，设置 `use_ragas_lib=true` 或 `--ragas-lib` 后调用真实 RAGAS 库指标
- 当前已补充 benchmark 服务和脚本，用于聚合离线质量、延迟和坏例诊断统计
- 前端主流程默认走业务 API；共享或生产环境需要在前端工作区中填写 `RAG_API_KEYS` 中的一枚 key（前端 UI 由独立仓库 `ifnodoraemon/nano-rag-ui` 维护）
- 默认建议本地使用 `MODEL_GATEWAY_MODE=mock`；接真实 provider 时再切到 `live`
- 直连外部 OpenAI-compatible provider
- 如果直连 Gemini API，请设置 `MODEL_GATEWAY_API_KEY`，覆盖 `GENERATION_MODEL_ALIAS / EMBEDDING_MODEL_ALIAS`，并通过 `DISABLE_RERANK=1` 跳过 rerank 阶段
- Docker Compose 默认启用 Bifrost；`docker compose -f docker/docker-compose.yml config --services` 应包含 `bifrost`

## 测试

```bash
cd nano-rag
pytest app/tests
```

前端验证在 [`ifnodoraemon/nano-rag-ui`](https://github.com/ifnodoraemon/nano-rag-ui) 仓库内完成（`npm run lint && npm run build`）。本仓库只在 Docker 构建时拉取上游打包，可通过：

```bash
docker build -f docker/frontend.Dockerfile \
  --build-arg UI_REF=<branch-or-commit> \
  -t nano-rag-frontend:test .
```

Docker 运行也属于交付要求。涉及后端、配置、依赖或部署行为的改动，提交前需要至少完成一次 Docker Compose 运行验证，或在交付说明里明确说明为什么本轮无法运行：

```bash
cd nano-rag
docker compose -f docker/docker-compose.yml up -d --build
curl http://127.0.0.1:8000/health
docker compose -f docker/docker-compose.yml down
```

GitHub Actions 会执行后端测试、Compose 配置校验，以及 app/frontend Docker image build；前端 lint/build 由上游 `nano-rag-ui` 仓库自身的 CI 负责。
如果要在 GitHub 上跑真实 Gemini smoke test，请在仓库 Secrets 配置 `GEMINI_API_KEY`，然后手动触发 `Live Smoke` workflow。
