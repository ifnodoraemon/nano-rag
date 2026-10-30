# 单模型 RAG 实现与 RAGFlow 对比

日期：2026-10-30。实现基于 Nano RAG `e63d405` 的工作区改动；RAGFlow 对比固定到 `a609e62d05d8d077d7698d3760dba2f1472d5c1c`（提交日期 2026-10-29），避免把历史版本的架构当作当前能力。

所有模型 ID 均来自 env。`GENERATION_MODEL_ALIAS` 无内置默认值；解析模型未单独填写时继承它，显式填写不同 ID 会拒绝启动。不同 API 协议或代理地址可以服务同一个模型，因此不以主机名判断是否同一模型。查询、读取规划、知识编译、问题支撑检查、证据审计和主张验证复用同一个生成客户端；没有增加 embedding、学习型 sparse encoder 或独立 reranker。

**本次落地范围**

| 上轮审查项 | 实现与验证路径 |
| --- | --- |
| 元数据过滤丢条文 | `wiki/search.py` 按父文档 provenance 过滤并保留 article；覆盖类型、日期、版本过滤 |
| 深读丢尾部证据 | `retrieval/artifacts.py` 按命中节点与词法相关度选取，再应用读取预算；折叠文档保留多个条文位置，页重复挤占时扩大抓取；上下文保留完整节点到摘取阶段 |
| 不真实的完整性标记 | `context_original_chars`、`context_truncated`、`document_read_complete` 和不可变 `EvidenceSet` 分别记录原文与节选覆盖 |
| 审计与生成看到不同内容 | `agentic/evidence_set.py` 固定同一份节选与图片字节；生成阶段禁止无记录地再次裁剪或重读变化后的图片；引用验证只针对可见证据，引用位置映射回原文 |
| 多文档被首篇挤占 | EvidenceDriller 按来源轮转分配名额，并按不同词项覆盖分配预算；重复背景不再累积权重 |
| 自然语言问题定位失败 | 问题、子目标和关键词变体共同进入证据定位；使用配置的词法分词，英文按词边界匹配，检查整个源段落 |
| 纠错候选不扩展 | 深度传入版本选择后的候选上限；每轮携带已读节点，补查可读取未读材料 |
| 旧版唯一命中被误当现行 | 版本选择查询完整来源目录；支持当前、历史时点、指定版本和版本对比 |
| 图扩展没有触发条件 | 统一查询计划输出 `requires_graph`，发现 trace 传给 auto 图扩展；图结果继续校验知识库和版本范围 |
| 延迟与成本统计不完整 | 请求结束后记录真实端到端耗时；ContextVar 归集并发调用、JSON 重试、HTTP 尝试、未知用量与每轮状态 |
| 单模型约束与默认值 | 移除源码、YAML、Compose、初始化脚本及示例中的具体模型默认值；初始化检查要求 env 指定模型 |
| 引用指标误报满分 | 增加数值—单位绑定检查；默认由同一 LLM 逐句校验实体、属性、值、单位和条件，引用原文锚点由程序校验 |

新增默认链路为：统一查询计划 → 分字段词法发现/版本选择 → 节点阅读计划 → 原文读取 → 固定证据快照 → 审计及有界补读 → 生成 → 同模型主张校验。

**可借鉴技术与实际落点**

| 能力 | Nano RAG 实现 |
| --- | --- |
| 小块命中、父级及邻接展开 | 条文页保存节点 ID、原文位置；`read_nodes` 可展开相邻片段和同源父节点的分段，始终受节点预算约束 |
| Contextual BM25 | 标题、章节路径、适用范围、源节点摘要进入独立 context 字段；没有引入 embedding |
| 文档树导航 | 阅读 manifest 提供路径、子分支、节点摘要与剩余节点；`navigate_tree` 支持分支下钻和分页 |
| 证据驱动多跳 | 查询子目标带依赖关系；审计输出带原文引用的中间事实、完成状态及结构化读取动作；每轮完整写入 trace |
| 段落 doc2query | 同模型按节点编译问题、答案引用及别名；先校验原文锚点，再同模型判断问题是否可由引用回答，丢弃不合格问题并记录数量 |
| 分字段词法检索 | 正文、标题、context、aliases、questions 分开统计计分；生成的问题不再计入正文长度；字段权重可通过 env 配置 |
| 显式引用图 | `follow_reference` 读取同文档引用；跨文档目标必须在源文本中明确出现其标题，且双方满足知识库、过滤和版本限制 |
| 表格查询与计算 | `query_table` 支持类型化筛选、行查询、count/sum/avg/min/max；Decimal 执行计算并保留所有参与行 ID，不执行模型生成的 SQL 或代码 |
| 调用预算与缓存 | 合并上下文改写、分解、scope 与查询扩展；查询计划缓存隔离知识库、会话和过滤条件；源 artifact 缓存随文件版本失效；请求有模型调用与检索轮数上限 |
| 树摘要与集合 Map-Reduce | 离线节点摘要沿结构树组织；`scan_collection` 分页记录覆盖，`summarize_collection` 用同模型 map 出带原文锚点的事实，再由正常合成阶段 reduce |
| 磁盘词法索引 | 可选 SQLite FTS5；每个进程拥有临时磁盘索引，持久 wiki 是重建来源。它使用 FTS5 原生多列 BM25，与内存公式分数不等价，定制 k1/b 或 bm25x 需使用内存后端 |

集合问题会区分文档、全文内容或表格行的覆盖；扫描不足时不得输出“已完整回答”的结构化标记。表格分页需累计实际可见行，聚合需向回答阶段提供确定性计算结果；不是发起一次表格查询就算读全。Map 生成的摘要是导航信息，最终可引用证据仍为原文或有来源行的确定性计算结果。若 map 已扫描全库，但最终证据预算容纳不了全部已提取事实，覆盖仍标记为不完整；当前 Map-Reduce 是受预算约束的实现。

**RAGFlow 当前代码给出的启发**

RAGFlow 当前也有纯 BM25 工具，不能把它概括为“只能向量检索”。其工具把关键词发现与权威原文深读分开：命中摘要用于筛选，再按 chunk ID 读取正文。这与本次保留命中位置的实现方向一致。[RAGFlow BM25 工具](https://github.com/infiniflow/ragflow/blob/a609e62d05d8d077d7698d3760dba2f1472d5c1c/internal/agentic_rag/tool_search_bm25_chunks.go)

其编译导航树用于按主题逐层定位文档。我们保留 BM25 的第一跳，并在候选文档内提供显式路径和分页导航；没有照搬全套服务。[RAGFlow 树导航](https://github.com/infiniflow/ragflow/blob/a609e62d05d8d077d7698d3760dba2f1472d5c1c/internal/agentic_rag/tool_navigate_tree.go)

其证据登记机制为模型真正看过的材料分配稳定引用身份。我们采用不可变 EvidenceSet，把可见节选、原文节点与引用范围绑定，避免仅凭“检索到过”就允许引用。[RAGFlow 证据登记](https://github.com/infiniflow/ragflow/blob/a609e62d05d8d077d7698d3760dba2f1472d5c1c/internal/agentic_rag/evidence_registry.go)

其知识库配置支持父子块、自动问题、增强上下文和全文字段选择。适合借鉴的是结构化加工与字段治理；依赖独立 embedding 或其他识别模型的方案超出我们的约束，未引入。[RAGFlow 数据集配置](https://github.com/infiniflow/ragflow/blob/a609e62d05d8d077d7698d3760dba2f1472d5c1c/docs/guides/dataset/configuration.md)

进一步值得借鉴的是知识维护与评测产品能力：将人工反馈回收为回归案例、以统一类型维护自定义元数据，以及原页/原文位置的可视化校对。这些需要围绕本项目的实际用户操作设计；本次提供的是检索与证据链路，并未复制 RAGFlow 的完整管理界面或编排平台。[元数据管理](https://github.com/infiniflow/ragflow/blob/a609e62d05d8d077d7698d3760dba2f1472d5c1c/docs/guides/dataset/metadata_management.md)

**配置与升级**

模型 ID 必须来自 env；`scripts/init_env.py` 会保留已有配置，但不再替用户选择模型。`DOCUMENT_PARSER_MODEL` 可留空继承；API 地址和协议仍由相应 env 指定。既有 `COMPOSE_*` 别名保留迁移兼容。

手动 Live Smoke CI 也改为读取仓库 vars/secrets 并在 Docker 中运行；移除了遗留的固定模型名和 embedding/vectorstore 配置。使用该工作流时需设置 `GENERATION_MODEL_ALIAS`、API 地址与凭据，以及所用解析协议的配置。

| env | 默认值 | 用途 |
| --- | --- | --- |
| `GENERATION_MODEL_ALIAS` | 无，必须填写 | 所有生成与推理阶段的模型 ID |
| `DOCUMENT_PARSER_MODEL` | 继承上述 ID | 解析能力的同模型配置；不同 ID 拒绝启动 |
| `RAG_QUERY_PLAN_ENABLED` | true | 统一查询计划 |
| `RAG_NODE_NAVIGATION_ENABLED` | true | 原文节点和结构导航 |
| `RAG_KNOWLEDGE_COMPILATION_ENABLED` | true | 摄入时生成带锚点的问题、别名和树摘要 |
| `RAG_CLAIM_VERIFICATION_ENABLED` | true | 同模型逐句主张校验 |
| `RAG_SCOPE_PREFERENCE_ENABLED` | false | 保留尚无稳定净收益的旧 scope 排序实验 |
| `RAG_LEXICAL_BACKEND` | memory | 可设为 sqlite；不增加检索模型 |
| `RAG_BM25_FIELD_WEIGHTS` | 空，使用内置通用权重 | JSON 对象，可配置 context、aliases、questions；只接受有限非负数 |
| `RAG_MAX_RETRIEVAL_LOOPS` | 4 | 含首次检索的总轮数 |
| `RAG_MAX_MODEL_CALLS` | 32 | 请求中的模型逻辑调用上限，包含 JSON 修正调用 |
| `RAG_QUERY_CACHE_ENTRIES` | 256 | 统一查询计划缓存容量；0 关闭 |
| `RAG_QUERY_CACHE_TTL_SECONDS` | 300 | 查询计划缓存有效期 |

模型和语义配置进入摄入指纹。旧语料需要重新摄入，才会生成新的段落问题、别名和树摘要；已有原文不会凭空获得这些派生字段。原有回归报告保留其基线意义，不能当作新版本测量值。

**验证口径**

回归测试覆盖过滤与条文、重复页扩大抓取、多条文位置、长段落尾部、证据预算与来源公平性、不可变文本/图片快照、引用原文位置、跨知识库读取、节点校验、完整版本目录、默认调用链、缓存、两轮补读、表格计算、集合 map、env 模型约束及无模型离线评测。全量结果为 **615 passed，5 个既有第三方警告**；ruff、diff 空白校验和两套 Compose 配置校验均通过。真实调用的数值与验证边界见[配套验证记录](ragflow-comparison-and-implementation-2026-10-30.validation.json)。

真实冒烟使用隔离 Docker 容器、现有 env 凭据与临时语料，没有更改运行中服务或其知识库。它验证真实模型接口和原文锚点衔接，不是大样本准确率或性能 A/B。首次真实运行暴露了提供方忽略原生 JSON Schema 的情况，已在新结构化阶段的提示中同时发送明确输出契约，并保留严格解析。

当前 SQLite + 知识编译冒烟通过：同一个 env 模型 ID，离线知识编译 2 次调用、2739 tokens；问答 5 次调用、6814 tokens，端到端 10.5596 秒，无重试。9514 字符原文中的答案位于 offset 9452；模型可见 3031 字符，完整性标记为 false，返回引用的原文起点正确为 9452。该答案通过原文锚点与语义主张校验。调试过程中还修复了英文分词定位、摘取窗口超预算、引用标签方括号语法及审计事实输出字段不一致；未知标签和无依据引用仍直接报错。

默认内存后端也通过真实冒烟：同一 env 模型 ID、5 次调用、6594 tokens、13.4198 秒，无重试；相同尾部答案及原文引用位置正确。该运行未执行离线知识编译，两组耗时不能作为后端性能优劣的对照。现有生产 Compose env 的生成与解析模型 ID 已只读核对为一致。代码已验证，运行中服务尚未重建，旧语料尚未重新摄入。

运行方式：

```bash
python -m pytest app/tests -q
python -m ruff check app scripts/init_env.py scripts/live_single_model_smoke.py scripts/run_retrieval_eval.py
# 在已构建本版代码的 Docker app 容器中执行；只写临时语料
python scripts/live_single_model_smoke.py --compile
```

`scripts/run_retrieval_eval.py` 默认是完全离线的确定性评测，不构造模型客户端；显式加 `--llm` 才启用模型辅助发现。引用数值检查只是确定性约束，同模型语义审计也有相关错误风险；新算法的通用收益仍需未参与调优的留出集和人工抽查验证。
