# Nano RAG

根据 `rag_project_plan_v2.md` 初始化的第一阶段企业级 RAG 项目骨架。

当前版本提供：

- `FastAPI` 服务，包含 `/health`、`/ingest`、`/chat`、`/retrieve/debug`、`/traces`
- 业务接入版 API，包含 `/v1/rag/chat`、`/v1/rag/ingest`、`/v1/rag/feedback`、`/v1/rag/traces/{trace_id}`、`/v1/rag/benchmark/run`
- `ingestion / retrieval / generation` 主链路模块拆分
- 统一 `model_client`，所有模型调用都走 OpenAI-compatible Gateway
- 默认 `mock gateway` 模式，可在没有真实模型 API 时先跑通本地自助迭代闭环
- `Milvus` 仓储抽象，默认走真实向量库，保留内存回退仅用于临时调试
- `Docker Compose` 骨架，包含 `app / milvus / litellm / phoenix`
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
uvicorn app.main:app --reload --app-dir .
```

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
- 当配置了 `RAG_API_KEYS` 后，业务 API 与高级调试 API 都会要求同一套 key
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

Compose 默认启用真实向量库：

```bash
VECTORSTORE_BACKEND=milvus
MILVUS_URI=http://milvus:19530
```

### 4. 准备模型网关配置

```bash
cp docker/litellm/.env.example docker/litellm/.env
```

按实际 provider 填写 `docker/litellm/.env`，然后调整 [models.yaml](/home/ifnodoraemon/myagent/nano-rag/configs/models.yaml)。

### 5. 直接使用 Gemini API

如果不走本地 LiteLLM，而是直接把服务指向 Gemini API 的 OpenAI-compatible endpoint，可设置：

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
- `/health` 现已同时兼容 LiteLLM 的 `/v1/models` 和 Gemini 的 `/models`
- 业务 API 中的 `kb_id` 已预留，但当前部署仅支持 `default`

### 6. 按能力拆分模型网关

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

## 示例

### ingest

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"path":"./data/raw"}'
```

### chat

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"差旅报销多久内提交？","top_k":10}'
```

### business chat

```bash
curl -X POST http://127.0.0.1:8000/v1/rag/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"差旅报销多久内提交？","kb_id":"default","tenant_id":"demo-tenant","session_id":"session-001"}'
```

如果已启用 `RAG_API_KEYS`，可加上：

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

- 默认向量仓储使用 `Milvus`
- 如需临时回退内存模式，可显式设置 `VECTORSTORE_BACKEND=memory`
- PDF 优先走 `Docling`，未安装时回退 `pypdf`
- `Phoenix` 已在运行环境中拉起，当前代码侧仍以本地 trace store 为最小实现
- `RAGAS` 相关脚本已接入离线评测入口，当前默认使用确定性聚合指标，后续可替换成真实 RAGAS 流程
- 当前已补充 benchmark 服务和脚本，用于聚合离线质量、延迟和坏例诊断统计
- 前端主流程默认走业务 API；如果配置了 `RAG_API_KEYS`，需要在前端工作区中填写对应 key
- 默认 `MODEL_GATEWAY_MODE=mock`，所以在没有真实 provider 时也能先跑通本地循环
- 切到真实模型时，将 `MODEL_GATEWAY_MODE=live` 并填好 `docker/litellm/.env`
- 如果直连 Gemini API，请设置 `MODEL_GATEWAY_API_KEY`，并通过 `DISABLE_RERANK=1` 跳过 rerank 阶段

## 测试

```bash
cd nano-rag
pytest app/tests
```
