# 企业级 RAG 系统项目规划（方案 C / Bifrost 主线 / Milvus 原生 Hybrid / RAGAS 库）

## 1. 文档目的

本文档基于方案 B（v2）和项目实际演进情况，修订技术选型与里程碑规划。

与方案 B 的关键差异：

- 模型网关从 **LiteLLM** 切换为 **Bifrost 默认主线**（LiteLLM 保留为可选 profile）
- Hybrid 检索已落到 **Milvus v2.5 原生 full-text search + hybrid search**，memory 后端保留 BM25 fallback
- 编排框架**去掉 LangChain**，沿用自研 pipeline（已验证更轻量可控）
- 评测从自研指标升级为 **RAGAS 库.x 标准库 + 自研指标补充**
- Rerank 从 disabled 改为 **网关可配置启用**
- 补全 **eval/replay** 功能

---

## 2. 项目现状总结

### 2.1 已完成（方案 B 第一阶段全部 + 额外能力）

| 能力 | 状态 |
|------|------|
| Docling 文档解析 | ✅ |
| FastAPI 服务 + 4 类 API | ✅ /health, /ingest, /chat, /retrieve/debug |
| Milvus 向量库 | ✅ |
| Phoenix tracing | ✅ |
| 统一模型客户端（GatewayClient 基类） | ✅ embeddings/rerank/generation |
| Model alias + per-capability gateway | ✅ |
| Mock 模式 | ✅ |
| Citation 返回 | ✅ |
| Semantic Chunker（按章节/句子切块） | ✅ 超出原计划 |
| Hybrid 检索 | ✅ Milvus 原生 hybrid + memory BM25 fallback |
| Query Rewrite / Multi-query / HyDE | ✅ 超出原计划 |
| 多租户（kb_id / tenant_id） | ✅ 超出原计划 |
| Freshness Filter | ✅ 超出原计划 |
| Metadata Rerank | ✅ 超出原计划 |
| Benchmark Service | ✅ 超出原计划 |
| Diagnostics Service | ✅ 超出原计划 |
| Feedback Store | ✅ 超出原计划 |
| Wiki Module | ✅ 超出原计划 |
| React 前端 SPA | ✅ 超出原计划 |
| API Key Auth | ✅ 超出原计划 |
| 31 个测试模块 | ✅ |
| Bifrost 默认服务 | ✅ 已在 docker-compose 中 |

### 2.2 剩余重点

| 项目 | 现状 | 目标 |
|------|------|------|
| Fallback/路由策略 | Bifrost 配置已存在 | 端到端验证 provider fallback 与日志可观测性 |
| 中文 analyzer 效果 | 原生 hybrid 已接入 | 用真实中文业务语料做效果校准 |

---

## 3. 技术方案修订

### 3.1 技术栈（修订后）

- 文档解析：Docling v2.90+
- 编排框架：**自研 pipeline**（不引入 LangChain）
- 向量数据库：Milvus v2.5+（原生支持 full-text search + hybrid search）
- 向量模型：Gemini Embedding（3072 维）或通过 Bifrost 路由的任意 embedding API
- 重排模型：通过 Bifrost 转发的 rerank API（配置 provider key 后启用）
- 生成模型：通过 Bifrost 转发的 LLM API
- 模型网关：**Bifrost**（默认），LiteLLM（可选 profile）
- tracing / 实验：Phoenix
- 离线评测：RAGAS 库.x + 自研指标
- API 服务：FastAPI
- 配置管理：YAML + 环境变量
- 容器编排：Docker Compose
- 前端：React 19 + TypeScript + Vite

### 3.2 为什么 Bifrost 优先而非 LiteLLM

| 维度 | Bifrost | LiteLLM |
|------|---------|---------|
| 性能 | 11µs overhead @ 5k RPS | ~8ms P95 @ 1k RPS |
| 语言 | Go（编译型，低内存） | Python |
| 已集成 | 项目已有 Bifrost profile | 需新增服务 |
| MCP 支持 | 原生 MCP Gateway | 需要 SDK 集成 |
| 语义缓存 | 内置 | 需额外配置 |
| 社区/生态 | 4k stars，较新 | 44k stars，更成熟 |
| Fallback | 原生 automatic fallback | 需配置 fallback |
| Embedding/Rerank | 通过 OpenAI-compatible 转发 | 原生支持 /embeddings, /rerank |

选择理由：
- 项目已集成 Bifrost，优先利用已有能力
- 性能优势在生产环境下有实际意义
- LiteLLM 保留为可选 profile，用于需要更成熟生态的场景（如 100+ provider 开箱即用）

### 3.3 为什么去掉 LangChain

项目自研 pipeline 已完成所有编排需求：
- ingestion / retrieval / generation 三个 pipeline 独立可控
- 引入 LangChain 会增加依赖链、版本耦合、调试复杂度
- LangChain 当前重心在 Agent/LangGraph，纯 RAG 场景收益小
- 自研 pipeline 在测试覆盖和可维护性上更优

### 3.4 为什么使用 Milvus 原生 Hybrid

Milvus 后端不再维护应用内 BM25 索引，原因是：
- 内存索引，不持久化，重启后需重建
- 不支持增量更新同步
- 分词对中文支持有限

Milvus v2.5 原生能力：
- 内置 BM25 full-text search（基于 analyzer + tokenization）
- 原生 hybrid search（dense + sparse 向量同时检索 + RRF 融合）
- 数据持久化，与向量数据同生命周期
- 支持 Milvus 的元数据过滤与多租户

当前收益：
- Milvus 后端直接使用 `AnnSearchRequest` 实现 dense + sparse 多路召回
- 减少 app 内存占用
- memory 后端保留 BM25 fallback，服务本地 mock、smoke 和单测

---

## 4. 修订后系统边界

### 4.1 输入类型（不变）

- `.pdf` / `.md` / `.txt` / `.html`

### 4.2 输出结构（不变）

```json
{
  "answer": "...",
  "citations": [{"chunk_id": "...", "source": "...", "score": 0.91}],
  "contexts": [{"chunk_id": "...", "text": "...", "source": "...", "score": 0.91}],
  "trace_id": "..."
}
```

### 4.3 运行方式

Docker Compose 启动：

- `app` 容器
- `milvus` (+ etcd + minio)
- `bifrost`（**默认启用**，不再是可选 profile）
- `phoenix`
- `frontend`

LiteLLM 作为 `litellm` profile 可选启用。

---

## 5. 修订后系统架构

### 5.1 离线数据流

```text
原始文件
-> 文件扫描
-> Docling 解析
-> 文本标准化
-> Semantic Chunker（按章节/句子）
-> Bifrost / embeddings
-> 写入 Milvus（dense 向量 + sparse 向量/full-text 字段）
-> 写入 metadata
```

### 5.2 在线调用流

```text
用户问题
-> query normalize / rewrite（可选）
-> Bifrost / embeddings
-> Milvus hybrid search（dense + full-text + RRF）
-> Bifrost / rerank
-> context builder + freshness filter
-> Bifrost / chat or responses
-> answer + citations
-> 写入 Phoenix trace
```

### 5.3 评测流

```text
线上 trace / 本地样本
-> 形成评测集（JSONL）
-> 运行 retrieval + generation pipeline
-> RAGAS 库.x 标准指标（faithfulness, answer_relevance, context_precision）
-> 自研指标（exact_match, context_recall, conflict/insufficiency 检测）
-> 输出分数与结果明细
```

### 5.4 Replay 流

```text
Phoenix trace_id
-> 读取 trace 详细数据
-> 重放 retrieval / rerank / generation 各阶段
-> 对比原始结果与重放结果
-> 输出 diff 报告
```

---

## 6. 修订后代码目录变化

### 6.1 需新增/修改的文件

```text
app/
  retrieval/
    milvus_hybrid.py         # 新增：基于 Milvus 原生 full-text + hybrid search
    bm25.py                  # 保留但标记 deprecated，后续移除
  eval/
    replay.py                # 重写：基于 Phoenix trace 的 replay
    # ragas lib integrated into ragas_runner.py       # 新增：RAGAS 库.x 集成
docker/
  bifrost/
    config.json              # 更新：添加 rerank/fallback 配置
docker-compose.yml           # 修改：bifrost 从可选 profile 改为默认服务
configs/
  models.yaml                # 更新：rerank default_alias 改为实际模型
```

### 6.2 保留的本地 fallback

```text
app/retrieval/bm25.py                 # memory 后端 fallback
app/retrieval/hybrid_retriever.py     # 统一 native Milvus 与 memory fallback 的入口
app/retrieval/hybrid_fusion.py        # memory 后端融合逻辑
```

---

## 7. 核心模块修订设计

## 7.1 model_client 模块（不变，但 Bifrost 成为默认路由）

现有 `GatewayClient` 基类 + 三客户端架构保持不变。
变化在于 `configs/models.yaml` 中 `model_gateway.base_url` 默认指向 Bifrost：

```yaml
model_gateway:
  base_url: ${MODEL_GATEWAY_BASE_URL:-http://bifrost:8080/v1}
  api_key: ${MODEL_GATEWAY_API_KEY:-change-me}
```

Bifrost 配置中设定 fallback：

```json
{
  "providers": {
    "primary": {
      "provider": "google",
      "apiKey": "env.GOOGLE_API_KEY"
    },
    "fallback": {
      "provider": "openai",
      "apiKey": "env.OPENAI_API_KEY"
    }
  }
}
```

## 7.2 retrieval 模块（已落地）

Milvus 后端使用原生 hybrid search：

1. Collection 新增 analyzer `text` 字段和 BM25 `sparse` 字段
2. 检索时用 `AnnSearchRequest` 发起 dense + sparse 双路召回
3. Milvus `WeightedRanker` 融合 dense / sparse 结果
4. memory 后端保留 BM25 fallback

关键 API：

```python
from pymilvus import AnnSearchRequest, WeightedRanker

dense_req = AnnSearchRequest(
    data=[query_vector],
    anns_field="vector",
    param={"metric_type": "COSINE"},
    limit=top_k,
)

sparse_req = AnnSearchRequest(
    data=[query_text],        # Milvus 会自动做 BM25 tokenize
    anns_field="text",        # full-text 字段
    param={"metric_type": "BM25"},
    limit=top_k,
)

results = collection.hybrid_search(
    [dense_req, sparse_req],
    ranker=WeightedRanker(0.7, 0.3),
    limit=top_k,
)
```

## 7.3 generation 模块（不变）

Prompt builder + service + answer_formatter 保持现有架构。

## 7.4 eval 模块（重大增强）

### 7.4.1 RAGAS 库集成

`ragas_runner.py` 直接依赖 `ragas` 库（>=0.4.3）：
- `run()` — 计算内置指标（exact_match, context_recall），不依赖 LLM
- `run_async()` — 调用 ragas 库 `evaluate()`，需要 LLM client

### 7.4.2 Replay 实现

```python
async def replay_trace(trace_id: str) -> dict:
    # 1. 从 Phoenix 读取 trace 各 span 数据
    # 2. 重放 retrieval 阶段
    # 3. 重放 rerank 阶段
    # 4. 重放 generation 阶段
    # 5. 对比原始 vs 重放结果
    # 6. 返回 diff 报告
```

## 7.5 tracing 模块（不变）

现有 Phoenix 接入保持不变。

---

## 8. Docker Compose 修订

bifrost 从可选 profile 改为默认服务：

```yaml
services:
  bifrost:
    image: maximhq/bifrost:v1.3.9
    container_name: nano-rag-bifrost
    # profiles: ["bifrost"]  # 删除此行，使其默认启动
    env_file: ./bifrost/.env
    ports:
      - "8080:8080"
    volumes:
      - ./bifrost/config.json:/app/data/config.json
    restart: unless-stopped

  # 可选 LiteLLM profile
  litellm:
    image: ghcr.io/berriai/litellm:main-stable
    profiles: ["litellm"]
    ports:
      - "4000:4000"
    volumes:
      - ./litellm/config.yaml:/app/config.yaml
```

---

## 9. 配置修订

### 9.1 models.yaml

```yaml
model_gateway:
  base_url: ${MODEL_GATEWAY_BASE_URL:-http://bifrost:8080/v1}
  api_key: ${MODEL_GATEWAY_API_KEY:-change-me}

embedding:
  default_alias: ${EMBEDDING_MODEL_ALIAS:-gemini/gemini-embedding-2}
  dimension: 3072
  base_url: ${EMBEDDING_API_BASE_URL:-}
  api_key: ${EMBEDDING_API_KEY:-}

rerank:
  default_alias: ${RERANK_MODEL_ALIAS:-cohere/rerank-v3.5}
  base_url: ${RERANK_API_BASE_URL:-}
  api_key: ${RERANK_API_KEY:-}

generation:
  default_alias: ${GENERATION_MODEL_ALIAS:-gemini/gemini-3.1-pro-preview}
  base_url: ${GENERATION_API_BASE_URL:-}
  api_key: ${GENERATION_API_KEY:-}

document_parser:
  enabled: ${DOCUMENT_PARSER_ENABLED:-true}
  default_alias: ${DOCUMENT_PARSER_MODEL:-gemini-3.1-pro-preview}
  base_url: ${DOCUMENT_PARSER_API_BASE_URL:-http://bifrost:8080/genai}
  api_key: ${DOCUMENT_PARSER_API_KEY:-}
```

### 9.2 settings.yaml

新增 hybrid search 配置：

```yaml
hybrid_search:
  enabled: ${RAG_HYBRID_SEARCH_ENABLED:-true}
  dense_weight: 0.7
  sparse_weight: 0.3
```

---

## 10. 修订后里程碑

### Milestone 1：最小闭环 ✅ 已完成

- ingest + retrieval + chat + citations + debug API
- 所有模型调用经 GatewayClient

### Milestone 2：Tracing ✅ 已完成

- Phoenix 接入
- 每次 chat 可看 trace

### Milestone 3：RAGAS 库 评测 + Replay ✅ 已完成

目标：
- 接入 RAGAS 库.x 标准库
- 保留自研指标作为补充
- 实现 eval/replay.py

完成标准：
- 能用 RAGAS 库.x 跑 faithfulness / answer_relevance / context_precision
- 自研指标仍可用
- 能基于 Phoenix trace replay 一次完整调用并输出 diff

**进度：replay 已实现，RAGAS 库.x 已接入 `run_async` / API `use_ragas_lib` / CLI `--ragas-lib` 路径**

### Milestone 4：Milvus 原生 Hybrid ✅ 已完成

目标：
- Milvus 后端使用 v2.5 原生 full-text search + hybrid search
- memory 后端保留本地 BM25 fallback，Milvus 后端不维护应用内 BM25 索引

完成标准：
- Collection 包含 analyzer text 字段、BM25 sparse 字段和 sparse index
- hybrid search 通过 `AnnSearchRequest` + `WeightedRanker` 实现
- Milvus 原生路径不再 bootstrap / update 应用内 BM25 index
- memory 后端继续支持本地 smoke / 单测

### Milestone 5：Bifrost 默认启用 + Rerank 支持 ✅ 已完成

目标：
- Bifrost 从可选 profile 改为默认服务
- Rerank 通过 Bifrost 支持；未配置 rerank provider key 的 Compose 环境默认关闭
- 配置 fallback 策略

完成标准：
- `docker-compose up` 默认启动 Bifrost ✅
- 所有模型调用默认经 Bifrost 转发 ✅
- Rerank 在配置 provider key 后执行 ✅
- Bifrost 配置中包含 primary + fallback provider ✅

### Milestone 6：模型路由策略验证

目标：
- 端到端验证多 provider fallback
- 不同环境可切不同 provider
- 验证成本/调用日志

完成标准：
- 主 provider 不可达时自动 fallback 到备选
- 通过 Bifrost Web UI 可查看调用日志
- 切换 provider 无需改业务代码

---

## 11. 开发顺序

1. Milestone 3：RAGAS 库 接入 + replay 实现（独立于基础设施变更）
2. Milestone 5：Bifrost 默认启用 + Rerank 支持（配置变更，风险低）
3. Milestone 4：Milvus 原生 Hybrid（已落地）
4. Milestone 6：Fallback 验证（剩余端到端验证项）

---

## 12. 关键工程原则（修订）

### 12.1 所有模型调用只走网关
禁止业务层直接调 provider SDK。Bifrost 作为默认网关。

### 12.2 所有模型名只认 alias
业务代码只用 alias，实际 provider model 在 Bifrost/LiteLLM 配置中映射。

### 12.3 检索和生成必须解耦
保持现有 pipeline 架构，检索问题与生成问题分开调试。

### 12.4 Hybrid 检索利用 Milvus 原生能力
不维护自建 BM25 索引，利用 Milvus v2.5 的 full-text search + hybrid search。

### 12.5 第一版就保留引用
已实现，继续维持。

### 12.6 第一版就做 debug 接口
已实现，继续维持。

### 12.7 评测采用双轨制
RAGAS 库.x 标准指标 + 自研业务指标并存，互为补充。

### 12.8 Bifrost 负责 provider 变更，应用负责业务稳定
应用层只消费稳定接口，不感知底层供应商切换。

---

## 13. 第二阶段预留点

代码结构保留口子，当前不做：

- `retrieval/multimodal.py` — 多模态检索
- `auth/acl.py` — ACL / 行级权限
- `ops/model_router_policy.py` — 复杂路由策略（灰度、A/B）
- `eval/replay.py` — ✅ 已实现
- `ingestion/incremental.py` — 增量索引更新

等以后补：
- 多模态页检索
- ACL
- 复杂路由策略 / 灰度切换
- provider 级成本审计
- 增量索引

---

## 14. 交付要求

代码智能体在修订后阶段必须交付：

1. ~~可运行的 Docker Compose 环境~~ ✅ 已有
2. ~~FastAPI 应用~~ ✅ 已有
3. ~~完整 ingest / retrieval / generation pipeline~~ ✅ 已有
4. ~~统一模型客户端层~~ ✅ 已有
5. ~~Milvus collection 初始化~~ ✅ 已有
6. **Bifrost 作为默认模型网关** ✅ 已完成
7. **Rerank 网关接入** ✅ 已完成（环境变量控制）
8. **RAGAS 库.x 集成** ✅ 已完成
9. **eval/replay.py 实际实现** ✅ 已完成
10. **Milvus 原生 hybrid search** ✅ 新 collection schema 与 `native_hybrid_search` 路径已实现
11. **Fallback 策略验证** ✅ 配置校验已纳入 CI；多 provider failover 通过手动 live smoke 执行
12. README 更新（启动、索引、问答、评测步骤）

### 明确要求

- 所有模型能力一律通过 Bifrost 网关调用
- 代码中不得直接硬编码 provider-specific SDK 调用
- 所有模型切换必须通过配置完成
- 所有关键流程必须可独立调试
- Hybrid 检索在 Milvus 后端利用原生能力；memory 后端保留自建 BM25 作为本地 fallback
- 评测双轨：RAGAS 标准 + 自研业务指标

---

## 15. 与方案 B 的差异总结

| 维度 | 方案 B | 方案 C（本版） |
|------|--------|----------------|
| 模型网关 | LiteLLM（必需） | **Bifrost（默认）**，LiteLLM 可选 |
| 编排框架 | LangChain | **自研 pipeline**（已验证） |
| Hybrid 检索 | 第一阶段不做 | **Milvus 原生 hybrid + memory BM25 fallback** |
| Query Rewrite | 第一阶段不做 | **已实现** |
| Rerank | Qwen3-Reranker（外部 API） | **可通过 Bifrost 转发启用** |
| 评测 | RAGAS 库 | **RAGAS 库.x + 自研指标双轨** |
| Replay | 未规划 | **本次补实现** |
| 多租户 | 第一阶段不做 | **已有 kb_id / tenant_id** |
| 前端 | 未规划 | **已有 React SPA** |

方案 C 比 B 更贴合项目实际演进，同时补齐了网关正式化、hybrid 检索持久化、评测标准化三个关键短板。

---

## 16. 全面对标（代码审查 2026-04-21）

### 16.1 代码规模

- 应用代码：~12,100 行 Python（不含 `__pycache__`）
- 测试模块：`app/tests` 全量通过
- 前端：React 19 + TypeScript SPA

### 16.2 API 端点对标

| API | 状态 | 位置 |
|-----|------|------|
| `GET /health` | ✅ | `main.py:123` |
| `POST /v1/rag/ingest` | ✅ path + upload | `routes_business.py:191` |
| `POST /v1/rag/ingest/upload` | ✅ 超计划 | `routes_business.py:221` |
| `POST /v1/rag/chat` | ✅ | `routes_business.py:160` |
| `POST /v1/rag/feedback` | ✅ 超计划 | `routes_business.py:322` |
| `GET /v1/rag/documents` | ✅ 超计划 | `routes_business.py:305` |
| `GET /v1/rag/traces/{id}` | ✅ | `routes_business.py:353` |
| `POST /v1/rag/benchmark/run` | ✅ 超计划 | `routes_business.py:374` |
| `POST /retrieve/debug` | ✅ | `routes_debug.py:58` |
| `GET /traces` | ✅ 分页 | `routes_debug.py:76` |
| `POST /eval/run` | ✅ | `routes_debug.py:142` |
| `POST /diagnose/trace` | ✅ 超计划 | `routes_debug.py:172` |
| `POST /diagnose/eval` | ✅ 超计划 | `routes_debug.py:196` |
| `POST /diagnose/auto` | ✅ 超计划 | `routes_debug.py:231` |
| `POST /replay/{trace_id}` | ✅ 本次实现 | `routes_debug.py` |
| `GET /debug/storage` | ✅ 超计划 | `routes_debug.py` |
| `GET /debug/parsed/{doc_id}` | ✅ 超计划 | `routes_debug.py` |

### 16.3 模块对标

| 模块 | 计划 v3 | 实际 | 差异 |
|------|---------|------|------|
| Ingestion | Docling + normalizer + chunker + embed + upsert | ✅ 完整 + semantic_chunker + rollback + artifact 持久化 | 超计划 |
| Model Client | GatewayClient + 3 客户端 | ✅ + DocumentParserClient + mock_gateway + per-capability routing | 超计划 |
| Vector Store | Milvus collection + CRUD | ✅ + InMemory 后备 + dynamic field + metadata filter + native hybrid schema | 当前 schema 已落地 |
| Retrieval | Vector search + rerank + context builder | ✅ + Milvus native hybrid / memory BM25 fallback + query rewrite + freshness + metadata rerank + wiki + filters | 需实测中文 analyzer 效果 |
| Generation | Prompt builder + LLM + formatter | ✅ + conflict notice + evidence summary + supporting claims | 完整 |
| Eval | RAGAS + 自研指标 | ✅ 自研指标默认 + RAGAS 库显式路径 | 真实 RAGAS 依赖可用评测 LLM |
| Replay | trace replay | ✅ 本次实现 | 完成 |
| Tracing | Phoenix OTEL | ✅ + local TraceStore + FeedbackStore | 完整 |
| Diagnostics | 未规划 | ✅ 392 行规则引擎 + AI 诊断 | 超计划 |

### 16.4 审查发现的代码问题与修复

| 问题 | 严重度 | 修复 |
|------|--------|------|
| `test_vector_repository.py::test_milvus_repository_refuses_to_drop_collection_on_dimension_mismatch` 因 pymilvus 未安装导致 ImportError 而非 RuntimeError | 中 | ✅ 已修复：monkeypatch pymilvus 模块 |
| `eval/replay.py` 为 stub | 高 | ✅ 已重写：完整 replay_trace 实现 + /replay API |
| Rerank 默认 disabled | 中 | ✅ 已修复：rerank 可通过环境变量和 provider key 启用 |
| Bifrost 为可选 profile | 中 | ✅ 已修复：删除 profiles，添加 fallback 配置，app depends_on bifrost |
| 无 LiteLLM 可选 profile | 低 | ✅ 已添加：litellm profile + config.yaml + .env.example |

### 16.5 剩余验证项

| 项 | 优先级 | 风险 | 状态 |
|----|--------|------|------|
| RAGAS 库.x 库集成 | P1 | 低 | 已接入 |
| Milvus 原生 hybrid search | P2 | 中（中文 analyzer 效果需真实语料验证） | 已接入 |
| Fallback 端到端验证 | P3 | 低 | 配置已校验；真实 failover 依赖第二 provider key |
