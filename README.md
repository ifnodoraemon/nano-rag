# Nano RAG

根据 `rag_project_plan_v2.md` 初始化的第一阶段企业级 RAG 项目骨架。

当前版本提供：

- `FastAPI` 服务，包含 `/health`、`/ingest`、`/chat`、`/retrieve/debug`、`/traces`
- `ingestion / retrieval / generation` 主链路模块拆分
- 统一 `model_client`，所有模型调用都走 OpenAI-compatible Gateway
- 默认 `mock gateway` 模式，可在没有真实模型 API 时先跑通本地自助迭代闭环
- `Milvus` 仓储抽象，默认带内存回退，方便本地调试
- `Docker Compose` 骨架，包含 `app / milvus / litellm / phoenix`
- 基础测试、trace 存储、配置模板和离线评测脚手架

## 目录

```text
nano-rag/
├─ docker/
├─ configs/
├─ data/
└─ app/
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

### 2. Docker Compose

```bash
cd nano-rag
docker compose -f docker/docker-compose.yml up -d --build
```

### 3. 准备模型网关配置

```bash
cp docker/litellm/.env.example docker/litellm/.env
```

按实际 provider 填写 `docker/litellm/.env`，然后调整 [models.yaml](/home/ifnodoraemon/myagent/nano-rag/configs/models.yaml)。

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
  -d '{"dataset_path":"data/samples/eval_sample.jsonl","output_path":"data/samples/eval_report_api.json"}'
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
  --dataset data/samples/eval_sample.jsonl \
  --output data/samples/eval_report.json
```

## 配置说明

- [settings.yaml](/home/ifnodoraemon/myagent/nano-rag/configs/settings.yaml): chunk、retrieval、timeout 等运行参数
- [models.yaml](/home/ifnodoraemon/myagent/nano-rag/configs/models.yaml): gateway 地址和模型 alias
- [prompts.yaml](/home/ifnodoraemon/myagent/nano-rag/configs/prompts.yaml): 生成提示词模板

## 现阶段说明

- 默认向量仓储支持内存模式，便于先跑通接口与测试
- 如果安装并配置 `Milvus`，可将 `VECTORSTORE_BACKEND=milvus`
- PDF 优先走 `Docling`，未安装时回退 `pypdf`
- `Phoenix` 已在运行环境中拉起，当前代码侧仍以本地 trace store 为最小实现
- `RAGAS` 相关脚本已接入离线评测入口，当前默认使用确定性聚合指标，后续可替换成真实 RAGAS 流程
- 默认 `MODEL_GATEWAY_MODE=mock`，所以在没有真实 provider 时也能先跑通本地循环
- 切到真实模型时，将 `MODEL_GATEWAY_MODE=live` 并填好 `docker/litellm/.env`

## 测试

```bash
cd nano-rag
pytest app/tests
```
