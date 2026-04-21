# 企业级 RAG 系统项目规划（方案 B / Qwen3 主线 / 外部 API 版）

## 1. 文档目的

本文档用于指导代码智能体或工程团队，从 0 开始实现一套可落地的第一阶段 RAG 系统。

本版本与上一版的关键差异：

- 所有模型能力均通过 **外部 API** 调用，不再自建 embedding / reranker / generation 模型服务。
- 系统中显式引入 **模型管理组件（Model Gateway / Model Router）**。
- 应用层不直接耦合具体厂商 API，而是统一通过模型管理层完成：
  - 模型别名管理
  - Provider 路由
  - 认证与密钥隔离
  - 重试 / 超时 / fallback
  - 成本与调用日志记录

本文档目标是提供一份可执行的工程规划，覆盖：

- 项目目标与边界
- 技术选型
- 系统架构
- 数据流与调用流
- 代码目录规划
- 模块拆分
- API 规划
- 配置规划
- 开发里程碑
- 交付标准
- 第二阶段演进预留

---

## 2. 项目目标

### 2.1 第一阶段目标

实现一条完整的文本 RAG 主线，支持：

1. 导入文档
2. 文档解析
3. 文本清洗与切块
4. 通过外部 embedding API 向量化
5. 向量检索
6. 通过外部 rerank API 重排
7. 基于上下文生成答案
8. 返回引用
9. 记录 tracing
10. 支持离线评测

### 2.2 第一阶段不做

第一阶段明确不做以下能力：

- 多租户业务权限模型
- ACL / 行级权限过滤
- 多模态检索主线
- hybrid sparse+dense 检索
- 增量索引更新
- 复杂 query rewrite
- 在线 A/B 实验平台
- 分布式大规模部署
- 自动计费与配额系统

### 2.3 第一阶段成功标准

满足以下条件则视为第一阶段完成：

- 能成功索引 PDF / Markdown / TXT / HTML 文档
- 能通过 API 发起问答
- 能返回 answer + citations + contexts
- 能通过调试接口看到检索结果与重排结果
- 能记录基本 tracing
- 能用一份 JSONL 测试集跑离线评测
- 能通过模型管理层切换 provider / model alias，而无需修改业务代码

---

## 3. 技术方案总览

### 3.1 技术栈

- 文档解析：Docling
- 编排框架：LangChain
- 向量数据库：Milvus
- 向量模型：Qwen3-Embedding-4B（外部 API）
- 重排模型：Qwen3-Reranker-4B（外部 API）
- 生成模型：外部 LLM API（由模型管理层统一转发）
- 模型管理组件：LiteLLM Gateway（推荐）
- tracing / 实验：Phoenix
- 离线评测：RAGAS
- API 服务：FastAPI
- 配置管理：YAML + 环境变量
- 容器编排：Docker Compose

### 3.2 为什么需要引入模型管理组件

当 embedding、reranker、generation 全部来自外部 API 时，如果应用直接分别对接多个 provider，会很快出现以下问题：

- 代码里散落多个 provider SDK
- 密钥管理分散
- timeout / retry / fallback 逻辑重复
- model name 与 provider 强耦合
- tracing、成本、日志口径不统一
- 切换供应商时改动面大

因此，建议在应用与外部模型 API 之间加入一层 **模型管理组件**。

### 3.3 模型管理组件的职责

模型管理组件不负责业务编排，只负责“统一模型接入”。职责包括：

- 统一 OpenAI-compatible API 入口
- model alias 到 provider model 的映射
- provider key 管理
- 重试、限流、fallback
- 超时与熔断策略
- 基础成本 / 调用日志记录
- 后续支持多 provider 灰度切换

### 3.4 模型管理组件选型

第一阶段推荐：

- **LiteLLM Gateway**

原因：

- 支持以统一接口接入 100+ LLM/provider
- 既支持 `/chat/completions`，也支持 `/embeddings`，并且项目说明已列出 `/rerank` 等端点
- 提供 Proxy/Gateway 模式，适合平台化接入
- 支持虚拟密钥、日志、预算、路由等能力

### 3.5 选型理由

#### Docling
用于文档解析，适合处理 PDF、DOCX、PPTX、HTML 等多类文档，便于后续扩展。

#### LangChain
用于组织 ingest pipeline、retrieval pipeline 和 generation pipeline，后续扩展 hybrid、多路召回、多模态支线时更容易。

#### Milvus
作为向量库，适合后续向量检索扩展，且对 RAG 场景成熟。

#### Qwen3-Embedding-4B
作为第一阶段文本向量模型主线。
选择原则：
- 直接采用较新的向量模型路线
- 兼顾效果与资源成本
- 作为后续评测与调优的基线模型

#### Qwen3-Reranker-4B
用于对召回候选进行重排，提高最终进入上下文的候选质量。

#### LiteLLM Gateway
用于统一接入所有外部模型 API，避免应用层直接绑定具体 provider。

#### Phoenix
用于 tracing、实验和后续回放调试。

#### RAGAS
用于离线评测，建立“可量化优化”的闭环。

---

## 4. 第一阶段系统边界

### 4.1 输入类型

第一阶段支持：

- `.pdf`
- `.md`
- `.txt`
- `.html`

### 4.2 输出结构

统一输出为：

```json
{
  "answer": "...",
  "citations": [
    {
      "chunk_id": "...",
      "source": "...",
      "score": 0.91
    }
  ],
  "contexts": [
    {
      "chunk_id": "...",
      "text": "...",
      "source": "...",
      "score": 0.91
    }
  ],
  "trace_id": "..."
}
```

### 4.3 运行方式

第一阶段采用单仓 monorepo 结构，配合 Docker Compose 启动：

- `app` 容器
- `milvus`
- `etcd`
- `minio`
- `litellm-gateway`
- `phoenix`

备注：
- 不再部署本地 embedding / reranker / generation 模型服务
- 所有模型调用均经 `litellm-gateway` 转发到外部 provider

---

## 5. 系统架构

### 5.1 离线数据流

```text
原始文件
-> 文件扫描
-> Docling 解析
-> 文本标准化
-> 切块
-> LiteLLM Gateway / embeddings
-> 写入 Milvus
-> 写入 metadata
```

### 5.2 在线调用流

```text
用户问题
-> query normalize
-> LiteLLM Gateway / embeddings
-> Milvus topK 检索
-> LiteLLM Gateway / rerank
-> context builder
-> LiteLLM Gateway / chat or responses
-> answer + citations
-> 写入 Phoenix trace
```

### 5.3 评测流

```text
线上 trace / 本地样本
-> 形成评测集
-> 运行 retrieval + generation pipeline
-> RAGAS 评测
-> 输出分数与结果明细
```

### 5.4 模型调用分层原则

代码必须遵循以下分层：

- 业务层：只知道“embedding / rerank / generate”能力
- 模型客户端层：只知道统一网关接口
- 模型网关层：负责 provider 路由与密钥管理
- 外部 provider：真正执行模型调用

禁止在业务代码里直接写：

- OpenAI SDK 调用
- 第三方 provider SDK 调用
- provider-specific model 名称硬编码
- provider-specific 重试逻辑

---

## 6. 服务拆分规划

### 6.1 app-service

职责：
- 对外暴露 HTTP API
- 提供 `/health`
- 提供 `/ingest`
- 提供 `/chat`
- 提供 `/retrieve/debug`
- 管理 pipeline 调用

第一阶段实现方式：
- FastAPI 单服务
- 内部拆分 ingestion / retrieval / generation 模块

### 6.2 model-gateway

职责：
- 统一转发外部模型请求
- 管理 provider keys
- 执行路由、超时、重试、fallback
- 统一暴露 OpenAI-compatible 接口

第一阶段实现方式：
- 采用 LiteLLM Proxy / Gateway

### 6.3 vector-store

职责：
- 管理 Milvus collection
- 执行 upsert / search
- 存储 chunk 向量及元数据

### 6.4 observability

职责：
- Phoenix tracing
- 评测结果留存
- 后续 experiment / replay 预留

---

## 7. 代码目录规划

```text
rag-system/
├─ docker/
│  ├─ docker-compose.yml
│  ├─ app.Dockerfile
│  └─ litellm/
│     ├─ config.yaml
│     └─ .env.example
├─ configs/
│  ├─ settings.yaml
│  ├─ models.yaml
│  └─ prompts.yaml
├─ data/
│  ├─ raw/
│  ├─ parsed/
│  └─ samples/
├─ app/
│  ├─ main.py
│  ├─ api/
│  │  ├─ routes_chat.py
│  │  ├─ routes_ingest.py
│  │  └─ routes_debug.py
│  ├─ core/
│  │  ├─ config.py
│  │  ├─ logging.py
│  │  ├─ tracing.py
│  │  └─ exceptions.py
│  ├─ ingestion/
│  │  ├─ loader.py
│  │  ├─ parser_docling.py
│  │  ├─ normalizer.py
│  │  ├─ chunker.py
│  │  └─ pipeline.py
│  ├─ model_client/
│  │  ├─ base.py
│  │  ├─ embeddings.py
│  │  ├─ rerank.py
│  │  ├─ generation.py
│  │  └─ schemas.py
│  ├─ vectorstore/
│  │  ├─ milvus_client.py
│  │  ├─ collections.py
│  │  └─ repository.py
│  ├─ retrieval/
│  │  ├─ retriever.py
│  │  ├─ reranker.py
│  │  ├─ context_builder.py
│  │  └─ pipeline.py
│  ├─ generation/
│  │  ├─ prompt_builder.py
│  │  ├─ service.py
│  │  └─ answer_formatter.py
│  ├─ eval/
│  │  ├─ dataset.py
│  │  ├─ ragas_runner.py
│  │  └─ replay.py
│  ├─ schemas/
│  │  ├─ document.py
│  │  ├─ chunk.py
│  │  ├─ chat.py
│  │  └─ trace.py
│  └─ tests/
│     ├─ test_chunker.py
│     ├─ test_retrieval.py
│     ├─ test_chat.py
│     └─ test_model_gateway.py
└─ README.md
```

---

## 8. 核心模块设计

## 8.1 ingestion 模块

职责：
把文件变成可检索 chunk。

输入：
- 文件路径
- 文档元数据

输出：
- `Document`
- `Chunk[]`
- `embedding[]`
- 入库结果

子模块：
- `loader.py`
  - 扫描文件
  - 按扩展名分流
- `parser_docling.py`
  - 调 Docling
  - 提取结构文本
- `normalizer.py`
  - 去空白
  - 去噪
  - 标准化换行
- `chunker.py`
  - 按段落/标题切块
  - token/字符上限控制
- `pipeline.py`
  - 串整个 ingest 流程

第一版 chunk 规则：
- chunk size：600–1000 tokens 等价字符范围
- overlap：80–150 tokens 等价字符范围
- 优先按标题/段落切
- 切不开再滑窗切

## 8.2 model_client 模块

职责：
统一调用模型管理层，而不是直接调 provider。

必须拆成三个客户端：
- `embeddings.py`
- `rerank.py`
- `generation.py`

接口原则：
- 业务层拿到的是统一 schema
- model alias 从配置读取
- 实际 base_url 指向 LiteLLM Gateway

接口示意：

```python
class EmbeddingClient:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

class RerankClient:
    def rerank(self, query: str, documents: list[str]) -> list[dict]:
        ...

class GenerationClient:
    def generate(self, messages: list[dict], model_alias: str | None = None) -> dict:
        ...
```

### 8.2.1 model registry 设计

应用内仍需保留一个轻量级 `model registry`，但它不是模型网关的替代品，而是应用侧配置映射层。

职责：
- 规定当前项目用哪些 alias
- 区分用途：embedding / rerank / generation
- 为不同环境切换 alias

例如：
- `embedding.default = qwen3-embed`
- `rerank.default = qwen3-rerank`
- `generation.default = main-chat-model`

## 8.3 vectorstore 模块

职责：
统一管理 Milvus collection 和 CRUD。

建议 collection 结构：

第一版先一个 collection：
- `chunks`

payload 元数据至少包括：
- chunk_id
- doc_id
- source
- title
- chunk_index
- text
- metadata_json

## 8.4 retrieval 模块

职责：
从 query 到可用上下文。

流程：
1. query normalize
2. embedding
3. Milvus topK 召回
4. reranker 重排
5. context 选择
6. 返回最终上下文

后续可扩展：
- hybrid retrieval
- parent-child retrieval
- 多路召回
- 多模态支线

## 8.5 generation 模块

职责：
根据上下文生成答案，并带引用返回。

子模块：
- `prompt_builder.py`
- `service.py`
- `answer_formatter.py`

输出结构建议：

```json
{
  "answer": "...",
  "citations": [
    {"chunk_id": "...", "source": "..."}
  ],
  "contexts": [...],
  "trace_id": "..."
}
```

Prompt 原则：
- 只能依据上下文回答
- 不能编造
- 证据不足就说无法确认
- 关键结论附引用

## 8.6 eval 模块

职责：
离线评测与回放。

第一版先做什么：
- 读一份 JSONL 评测集
- 对每个 query 跑一次完整链路
- 保存 answer / citations / retrieved_contexts
- 调 RAGAS 跑分

第一版评测集格式建议：

```json
{
  "query": "...",
  "reference_answer": "...",
  "reference_contexts": ["..."],
  "metadata": {}
}
```

## 8.7 tracing 模块

职责：
把一次调用的关键数据打到 Phoenix。

每次请求至少记录：
- query
- retrieved chunk ids
- rerank 后 ids
- final contexts
- prompt version
- model alias
- answer
- latency
- provider route metadata（可选）

---

## 9. 数据模型规划

### 9.1 Document

```python
class Document(BaseModel):
    doc_id: str
    source_path: str
    title: str
    content: str
    metadata: dict
```

### 9.2 Chunk

```python
class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    source_path: str
    title: str | None = None
    metadata: dict = {}
```

### 9.3 ChatResponse

```python
class Citation(BaseModel):
    chunk_id: str
    source: str
    score: float | None = None

class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    contexts: list[dict]
    trace_id: str | None = None
```

---

## 10. API 规划

第一版就 4 类接口。

### 10.1 健康检查
`GET /health`

### 10.2 建索引
`POST /ingest`

请求：
```json
{
  "path": "/data/raw"
}
```

### 10.3 问答
`POST /chat`

请求：
```json
{
  "query": "差旅报销多久内提交？",
  "top_k": 10
}
```

### 10.4 检索调试
`POST /retrieve/debug`

返回：
- 原 query
- topK 召回
- rerank 后结果
- 最终上下文

---

## 11. 配置规划

所有“会变”的东西都不要硬编码。

### 11.1 `models.yaml`

```yaml
model_gateway:
  base_url: http://litellm-gateway:4000
  api_key: ${LITELLM_MASTER_KEY}

embedding:
  default_alias: qwen3-embed

rerank:
  default_alias: qwen3-rerank

generation:
  default_alias: main-chat-model
```

### 11.2 `settings.yaml`

```yaml
chunk:
  size: 800
  overlap: 120

retrieval:
  top_k: 20
  rerank_top_k: 10
  final_contexts: 6

prompt:
  version: v1

timeout:
  embeddings_seconds: 30
  rerank_seconds: 30
  generation_seconds: 60
```

### 11.3 LiteLLM 网关配置示意

```yaml
model_list:
  - model_name: qwen3-embed
    litellm_params:
      model: openai/qwen3-embedding-4b
      api_base: ${EMBED_PROVIDER_BASE_URL}
      api_key: ${EMBED_PROVIDER_API_KEY}

  - model_name: qwen3-rerank
    litellm_params:
      model: openai/qwen3-reranker-4b
      api_base: ${RERANK_PROVIDER_BASE_URL}
      api_key: ${RERANK_PROVIDER_API_KEY}

  - model_name: main-chat-model
    litellm_params:
      model: openai/your-chat-model
      api_base: ${GEN_PROVIDER_BASE_URL}
      api_key: ${GEN_PROVIDER_API_KEY}
```

说明：
- 实际 provider 名称需按真实供应商与 LiteLLM 支持格式调整
- alias 对应用层保持稳定，provider model 可在网关层替换

---

## 12. 开发里程碑

### Milestone 1：最小闭环
目标：
跑通：
- ingest
- retrieval
- chat

完成标准：
- 能索引 10 个文档
- 能回答问题
- 返回引用
- 有 `/retrieve/debug`
- 所有模型调用经模型网关完成

### Milestone 2：加 tracing
目标：
接 Phoenix

完成标准：
- 每次 chat 都能看到 trace
- 能看到检索上下文和答案
- 能看到模型 alias / provider route 基础信息

### Milestone 3：加离线评测
目标：
接 RAGAS

完成标准：
- 有 30–50 条样本集
- 能批量跑一版评测
- 能输出结果文件

### Milestone 4：加模型路由策略
目标：
把模型管理层真正用起来

完成标准：
- 不同环境可切不同 provider
- generation 模型支持 fallback
- embedding / rerank alias 可独立替换

---

## 13. 第一阶段开发顺序

最合理的顺序是：

1. 先搭目录和配置骨架
2. 起 Docker Compose：Milvus + LiteLLM + Phoenix + App
3. 实现 model_client 三个客户端
4. 实现 ingest pipeline
5. 实现 retrieval pipeline
6. 实现 generation pipeline
7. 接 FastAPI：`/ingest`、`/chat`、`/retrieve/debug`
8. 接 Phoenix tracing
9. 补 RAGAS 评测

---

## 14. 关键工程原则

### 14.1 所有模型调用只走网关
禁止业务层直接调 provider SDK。

### 14.2 所有模型名只认 alias
业务代码只用 alias，例如：
- `qwen3-embed`
- `qwen3-rerank`
- `main-chat-model`

### 14.3 检索和生成必须解耦
很多问题根本不是生成错，而是检索错。

### 14.4 第一版就保留引用
不然你后面很难 debug 答案来源。

### 14.5 第一版就做 debug 接口
这个会极大提升开发效率。

### 14.6 网关负责 provider 变更，应用负责业务稳定
应用层只消费稳定接口，不感知底层供应商切换。

---

## 15. 第二阶段预留点

现在先不做，但代码结构要给它留口子：

- `retrieval/hybrid.py`
- `retrieval/multimodal.py`
- `auth/acl.py`
- `ops/model_router_policy.py`
- `eval/replay.py`

等以后补：
- BM25 / hybrid 检索
- 多模态页检索
- ACL
- 回放
- provider 级成本审计
- 复杂路由策略
- 灰度切换

---

## 16. 交付要求（给代码智能体）

代码智能体在第一阶段必须交付：

1. 一套可运行的 Docker Compose 环境
2. 一个 FastAPI 应用
3. 一条完整的 ingest pipeline
4. 一条完整的 retrieval pipeline
5. 一条完整的 generation pipeline
6. 一个统一模型客户端层
7. Milvus collection 初始化逻辑
8. LiteLLM 网关配置模板
9. Phoenix tracing 接入
10. RAGAS 评测脚本
11. README：包含启动、索引、问答、评测步骤

### 明确要求

- 所有模型能力一律通过外部 API 调用
- 代码中不得直接硬编码 provider-specific SDK 调用
- 所有模型切换必须通过配置完成
- 所有关键流程必须可独立调试
- 代码优先保证“先跑通”，不是先做复杂抽象

---

## 17. 最终建议

对于当前阶段，**需要引入模型管理组件**。

理由不是“为了架构好看”，而是因为你的模型能力已全部外部化；如果不加这一层，应用代码会过早和 provider 细节耦合，后续切模型、切供应商、加 fallback、做成本统计时都会反复返工。

因此，第一阶段推荐架构为：

- App（FastAPI + LangChain）
- LiteLLM Gateway（模型管理层）
- Milvus（向量库）
- Phoenix（tracing / experiments）
- RAGAS（离线评测）

这版方案比“直接在应用里调三个外部模型 API”更稳，也更适合交给代码智能体一次性搭出可演进骨架。
