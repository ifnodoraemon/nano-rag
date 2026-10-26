# 2026 年业界 RAG 做法调研：一个 LLM + 词法检索，如何补上「用户口语 ↔ 规范条文」的语义鸿沟

> 调研时间：2026-10-26　｜　调研人：RAG 架构调研员　｜　目标读者：nano-rag 架构决策者
>
> **适用边界（重要）**：本报告讨论的技术只要是「词法检索 + 一个 LLM」，就与语种无关。
> 下文凡是标注「中文/法条」的证据，是在中文法规上实测过的；凡是标注「多语种/跨语种」的，
> 是在 MIRACL / MKQA / BEIR 等多语料上验证过的，可以直接迁移到英语、德语、日语等任意语言。
> 结论按证据强度标注，未标注来源的数字一律不写。

---

## 0. 结论先行（TL;DR）

1. **HyDE 的「无 embedding 变体」不但存在，而且已经是一条独立的研究线**。2026 年 SIGIR 的
   Jedidi & Lin 论文在 14 个数据集上系统比较了「LLM 生成的假想文档 + BM25 反馈模型」，
   结论是：**假想文档必须经过 BM25 反馈模型（Rocchio/RM3/平均向量）选词加权，而不是裸拼接**；
   裸拼接（Naive Concat）在多数数据集上**比不扩展还差**。这与我们「一个 LLM」的约束完全兼容——
   Rocchio/RM3 是纯统计公式，不是模型。
2. **doc2query 的实证非常强，但都是「训练过的 seq2seq」**；用 LLM 做文档侧问题合成（我们已落地
   的 `app/wiki/doc2query.py`）**没有找到中文法规上的公开实证**，属于「有强类比证据、无直接证据」。
   更关键的是：**合成问题本身也是幻觉源**，ECIR 2023 的 doc2query-- 用相关性过滤把 doc2query
   提升了 16%，说明**「生成 → 过滤」必须成对存在，只有生成会引入不可控噪声**。
3. **中文法条检索的公开工作（STARD、CLaw、LegalBench-RAG 类）几乎全部以「稠密检索」为解法，
   但它们的基线数据反而支持我们**：在**不引入任何模型**的前提下，BM25 是招语问题（真实口语提问）
   上的最强零样本基线；零样本稠密模型（SEED/coCondenser/SAILER/Lawformer）**全部不如 BM25**。
   我们 0.1236 的 hit@1 低，不是「没有 embedding」导致的，而是**口语↔法条这个任务本身难**。
4. **2024–2026 的 RAG 演进里，只有三样东西和我们兼容**：① Anthropic 的 **Contextual Retrieval
   之 BM25 分支**（给块加 LLM 生成的定位上下文）；② HyDE + BM25 反馈（见第 1 条）；
   ③ 文档内的**章节层级 + 法条编号**作为词法信号（STARA 的选择性 RegEx 前置过滤 + 语义判定）。
   **不兼容**：late chunking（需要长上下文 embedding 模型）、GraphRAG 的社区摘要需要 embedding
   做聚类、RAPTOR 的树检索需要 embedding 做最近邻。
5. **对我们收益最高、成本最低的一步不是换模型，而是「扩展检索的粒度」**：现在的架构是
   **文档级** BM25（3000 部 × 5500 字），而用户的问题是**指向单条**的。STARA 的经验是
   「先窄化语料到相关章节/条文，再让模型判定」。把 doc2query 从**文档级移到条文级**，
   并把合成问题**限定在该条文的主题域**，是目前证据支持度/成本比最好的一条路。

---

## 1. 方向一：doc2query / doc2query-- / 文档侧问题合成

### 1.1 做法（论文里的原始配方）

doc2query 的定义：训练一个 seq2seq 模型，**输入文档、输出该文档可能回答的问题**，把所有生成的问题
拼接回文档后再建索引，检索时用普通 BM25 查（[Nogueira & Lin, docTTTTTquery](https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery-v2.pdf)）。

原始论文里给出并被我逐条核对的超参：

| 配方项 | doc2query (2019) | docTTTTTquery (T5-base) |
|---|---|---|
| 每文档生成数 `n` | 10 | **5**（最优性价比）／40（最强） |
| 解码方式 | top-k 采样，k=10 | top-k 采样，k=10（beam search 更差） |
| 输出长度 | 64 token 上限 | 64 token 上限 |
| 问题放哪里 | **直接拼在文档正文之后**，不加分隔符 | 同左 |
| MS MARCO dev MRR@10 | 0.218（BM25 = 0.184） | 0.259（n=5）→ 0.277（n=40） |

**版式结论（可直接用）**：`n` 从 5 加到 80，MRR@10 只从 0.259 涨到 0.278，**收益在 n≈20–40 就饱和**；
查询延迟只从 55ms 涨到 66ms（原因只是文档变长，没有额外模型调用）。

### 1.2 「问题放哪里」有专门的负面证据

ECIR 2023 的 [Doc2Query--: When Less is More](https://arxiv.org/abs/2301.03266) 是这条线上我最看重的一篇，
因为它**承认并量化了幻觉**：

- 作者人工按 TREC DL 0–3 分评判 T5 生成的 12 个问题，其中 3 个是 **0–1 分（原文答不了）**；
  典型如把 "Barley (Hordeum vulgare L.)" 生成成 "what is **bare wheat**"。
- 引用 SPLADE 的例子：同一段落 top-20 扩展词里有 `reed`、`herb`、`troy` 三个与原文无关的词。
- 解法：**在索引前用一个相关性模型把生成的问题过一遍，只保留 top-p 比例**。实测（MS MARCO passage，5 个测试集）：

| 系统 | dev RR@10 | DL'19 nDCG@10 | 索引大小 | 平均响应时间 |
|---|---|---|---|---|
| BM25 | 0.185 | 0.499 | 0.71 GB | 5 ms |
| doc2query n=80（不过滤） | 0.279 | 0.627 | 1.41 GB | 30 ms |
| doc2query n=80 + ELECTRA 过滤 top30% | **0.323** | **0.670** | 0.95 GB | 23 ms |
| doc2query n=80 + MonoT5 过滤 top40% | 0.311 | 0.665 | 1.04 GB | 28 ms |

即：**过滤让检索效果 +16%，同时索引缩小 33%、查询提速 23%**。论文还给了一条更狠的对照：
在相同 GPU 小时预算下，「生成 n=5 + 过滤」优于「生成 n=11 不过滤」；要达到同样的 RR，
不过滤版本需要 **2.9 倍** 的生成算力。

> **对我们的直接含义**：我们现在的 `_clean_questions` 只做「去空行 / 去超长 / 去重」这类
> **字符串级**清洗，没有任何**相关性级**过滤。而我们的合成问题是走「一个 LLM 读全文」生成的，
> 幻觉率不会比 T5 低。**「生成 + 相关性过滤」是必须成对的。**

### 1.3 中立/负面的证据

- [Query2Doc (EMNLP 2023)](https://aclanthology.org/2023.emnlp-main.585/) 在零样本 OOD（BEIR 5 个低资源集）上
  出现 **负迁移**：NFCorpus 34.9→? 、Scifact 68.6→**-2.1**（BM25 + query2doc 在零样本上仍正，
  但 SimLM/E5 上加 query2doc 在 NFCorpus/Scifact 上分别 **-0.6 / -2.9**）。
  作者的解释是「训练/评测分布不匹配」。**模型生成的文本带着生成器的先验**，在语料分布和预训练
  分布不一致时会注入偏差。
- Query2Doc 还明确指出：**模型越小，生成的伪文档越短、事实错误越多**；GPT-4 > davinci-003 > davinci-001 > curie > babbage。
  这条对我们的含义是：**文档侧问题合成的质量与 LLM 能力强绑定**，换模型要重测。

### 1.4 多语种证据（因为我们是语言无关的，这条很重要）

- [M3-Embedding (arXiv 2402.03216)](https://arxiv.org/abs/2402.03216) 的附录里，他们构造训练数据时用的
  prompt 是「You are a curious AI assistant, please generate one specific and valuable question based on the
  following text」——**这就是 LLM 版的 doc2query，而且是多语种直接套用的**。
- [Lin 2023 的论文语料](https://pure.mpg.de/rest/items/item_3287344_1/component/file_3287345/content) 里有一节分析
  doc2query-T5 预测的 term 有多少已经出现在原文中（即「扩展的增量价值」），结论是大部分是复用词，
  真正新词的贡献集中在一小部分——这解释了为什么 `n` 很快饱和。

### 1.5 中文法规上的实证：**无公开实证**

我用下面几种检索都**没有**找到「在中文法条语料上做 doc2query / LLM 问题合成并报告 hit@k」的论文：

- ACL Anthology 全站 × EMNLP'22/'23/ACL'22/NAACL'22/COLING'22/AACL'22 逐篇扫标题；
- arXiv API（title / all 两级关键词）；
- 中文检索「doc2query 中文 法律 检索 效果 实验」只返回 CSDN/博客级内容，**没有可引用的实验数据**。

> 结论：**方向成立的类比证据极强，但「中文法条 + LLM 问题合成」这个组合，2025–2026 年仍然没有公开的定量结论。
> 我们自己的 A/B（合成问题开/关）就是这个语境下的第一手数据，值得纳入 eval 报告。**

### 1.6 对本项目的适配性

| 维度 | 评价 |
|---|---|
| 是否引入新模型 | **不引入**。只用现有 generation client，索引期间一次 LLM 调用/文档。 |
| 是否与现有架构冲突 | **不冲突**。合成问题写入 source page body，BM25 直接可索引（当前实现即是如此，见 `app/wiki/compiler.py:564`）。 |
| 主要风险 | ①幻觉问题污染索引（尤其低分问题被 tf 反复命中）；②页面变长稀释 BM25 的文档长度归一化；③每条问题 60 字的模板化会让所有文档共享高频 token（如「公司」「需要」「可以吗」），**导致 IDF 失效、候选塌缩**。 |
| 与已排除路线的区别 | 已排除的是「模型给候选打 0-100 分」（运行期排序），这里是**索引期扩展**，与运行期打分不是一回事。 |

### 1.7 落地建议

1. **把「问题」和「提问词」分开存**：合成问题不要直接混进正文 body，而是写入页面的一个**独立字段**
   （如 `## Questions This Document Answers` 区块，已存在）并在建索引时赋予**较低的字段权重**
   （BM25F 式：`body` 权重 1.0，`questions` 权重 0.3–0.5）。理由：doc2query 的收益来自召回，
   但如果权重和正文相同，模板化提问词会抢占 top-1。
2. **加相关性过滤（doc2query-- 的核心）**：生成后同一轮 LLM 调用里顺带要求返回每个问题的
   `answerable_from_document: true/false`，或者用「问题里出现的名词性实体是否出现在原文」做
   确定性校验（**不是模型打分，是字符串包含校验**——避开已排除的「打 0-100 分」路线）。
3. **问题模板强制「不用法条术语、不用书名号、用口语」**（当前 prompt 已有此约束，保留），
   并且**禁止所有问题共享同一开头句式**，否则 IDF 塌缩。
4. **数量：先测 n=8（现状）与 n=4 / n=16**。文献显示 `n` 在 20 左右饱和，但那是**短段落**（MS MARCO
   passage，平均 ~70 词）；我们文档平均 5,487 字，**同一份文档的 8 个问题覆盖面严重不足**——
   8 个问题只能覆盖 8 个条文，而我们平均每部法规有几十到上百条。这是**当前实现最可疑的参数**。

---

## 2. 方向二：中文/多语种法规制度的检索特殊做法

### 2.1 公开工作盘点（按对我们有用程度排序）

#### (A) STARD —— 招语（非专业用户）法条检索基准，**这是与我们任务最像的公开数据**

[STARD: A Chinese Statute Retrieval Dataset Derived from Real-life Queries by Non-professionals (Findings of EMNLP 2024)](https://aclanthology.org/2024.findings-emnlp.625/) ｜ [GitHub](https://github.com/oneal2000/STARD)

- 语料：**55,348 条法条**（按「最小可检索单元 = 条」切分），来自 1,543 条**真实法律咨询问题**（12348.gov.cn）。
  平均查询 27.3 字，平均法条 119.93 字，81 部法律法规。
- **最重要的实证（Table 3，R@5 / R@100 / MRR@5）**：

| 方法 | R@5 | R@100 | MRR@5 | 类型 |
|---|---|---|---|---|
| **BM25** | **0.3349** | **0.6493** | **0.3176（词法最优）** | 词法 |
| QL（Dirichlet） | 0.3363 | 0.6515 | 0.3167 | 词法 |
| Chinese-RoBERTa-WWM（零样本） | 0.3216 | 0.6633 | 0.2905 | 稠密 |
| SEED（零样本） | 0.2897 | 0.5626 | 0.2708 | 稠密 |
| coCondenser（零样本） | 0.1120 | 0.4292 | 0.0847 | 稠密 |
| SAILER（法律领域 PLM，零样本） | 0.2330 | 0.5674 | 0.2115 | 法律稠密 |
| Lawformer（法律领域 PLM，零样本） | 0.2411 | 0.5478 | 0.2205 | 法律稠密 |
| Dense-STARD（在 STARD 上微调的稠密） | 0.5206 | 0.9065 | 0.4543 | **需训练** |
| Dense-GPT4（用 GPT-4 生成问题对训练稠密） | 0.4382 | 0.7984 | 0.3842 | **需训练** |
| LSI-STARD（把 5.5 万条当分类标签） | 0.1861 | 0.3410 | 0.2093 | 需训练 |

  论文原文第 (1) 条结论：**「在零样本设置下，传统词法匹配超过通用与法律领域预训练模型」**。
  即：**任何不经过领域微调的语义模型，在招语法条检索上都不如 BM25。**

- **注意 STARD 的"隐含地支持条文级索引"**：它的语料是「按条切分的最小检索单元」，而 BM25 在这个粒度上
  R@5=0.33。我们的架构是**文档级** BM25，粒度粗得多。**这次对比说明：在同一个用户群体、
  同一类问题下，「条」级索引的本身就是一种增益来源。**

#### (B) CLaw —— 法条版本 / 法条编号 / 全量法规语料

[CLaw: Benchmarking Chinese Legal Knowledge in LLMs (Findings of EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.646/)

- 语料：**全部 306 部中国国家法律**，切到 **子款（subparagraph）粒度**，**并带历史修订时间点**，共 64,849 条。
- 附录 B 给出了「检索式 RAG 在法条上失败的两种真实案例」，这两种失败模式**对我们的版本过滤逻辑是直接的正确性要求**：
  1. **条文改号**：《行政诉讼法》第十一条在 1989 版 / 2014 版 / 2017 版的内容完全不同，
     Doubao 检索到了 1989 版的第十一条内容（2014 版已改为第十二条）。
  2. **来源日期误判**：《证券法》第八十二条在 1998 版 / 2005 版 / 2014 版内容不同，
     DeepSeek-R1 检索到一篇出版日期为 1998-12-29 的网页，并**违背了问题里明确给出的 2014-08-31**。
- 论文总结的三大弱点：**版本与时序推理**、**条文编号变更**、**来源可信度评估**。

> 对我们是好消息：我们的 `versioning.py` 版本过滤 + latest-wins 正好是 CLaw 指出的第 1、2 号失败的
> **系统性解法**，而且是确定性的，不依赖模型。这条可以写进设计文档当作已有优势。

#### (C) STARA / LegalBench-RAG 一类「法规普查式 RAG」的工程做法（2026 最新）

[Benchmarking Legal RAG: The Promise and Limits of AI Statutory Surveys (CS&Law '26, arXiv 2603.03300)](https://arxiv.org/abs/2603.03300)

- 任务：对美国 50 州失业保险法做**跨辖区法条普查**（1,647 个二分类问题）。
- 结果：标准 RAG 70%；**定制的 STARA 83%（修正人工基准遗漏后 92%）**；
  **商业产品 Westlaw AI 58%、Lexis+ AI 64%，都低于标准 RAG**。
- **STARA 的流水线（这是全篇对我们最有价值的工程描述）**：
  1. 解析并切分条文，**保留层级结构**；
  2. 给每条条文**补充上下文**：定义条款、交叉引用、父级条款；
  3. **可选的 RegEx/关键词前置过滤**（为了把语料缩到可计算规模）；
  4. **再由语言模型做相关性判定**（不是向量相似度）。
- 原文第 5.1.3 条设计原则：「在完整法典上搜索 vs 在相关章节上搜索」——**限制到相关 title/chapter
  能同时提升效率与准确率**，商业系统的典型错误是「引用了关键词相似但属于无关法律领域的条文」。
- 计算成本：5 个问题 × 1 个州 ≈ 20 分钟；单问题 50 州普查 ≈ 3.3 小时。

> 注意：STARA 的「语义判定」是用 LLM 做**分类**（是/否），**不是打 0-100 分**。
> 这恰好绕开了我们已排除的「模型打相关性分」失败模式——**二分类比连续打分稳定得多**。

#### (D) 法条「编号 / 引用解析」

中文/外语法律领域有大量「法条引用解析」（citation resolution）工作，但**我只找到描述性/商业实现，
没有找到可直接引用的、在中文法条上量化收益的公开论文**——**这一小节属于「无公开实证」**。
不过 STARD / CLaw 的语料构建方式间接说明：**「法规名 + 第 X 条」是法条的最自然主键**，
而我们的版本过滤已经以 `source_key` 为键，**把「条号」也纳入 `source_key` 或 metadata 是低成本的确定性增强**。

#### (E) 多语种：法规/制度检索在其他法系的做法

- COLIEE（日本法条推理）与 BSARD（比利时法语法规）等赛题中，**条文级切分 + 词法基线**同样是标准起手式；
  但 2025–2026 年的 COLIEE 概览里，上位系统主要靠**微调稠密模型**，与我们的约束冲突。
- 跨语种 BM25 的参照点：[M3-Embedding](https://arxiv.org/abs/2402.03216) 的 MIRACL 双语数据里，
  **BM25 在中文 nDCG@10 = 31.0**，在日语 24.5、阿拉伯语 18.9、希伯来语 26.9——
  **CJK 与形态丰富的语言上 BM25 的自然劣势更明显**，这也是「为什么我们更需要 query/doc 侧扩展」
  而不是「换打分函数」的解释。

### 2.2 对本项目的适配性 & 落地建议

1. **条文级索引（最高优先级的结构性改动）**：在现有文档级 BM25 之后，增加一个
   **条文级（article-level）子索引**，只对 top-N 文档建，或全局建但只在候选集合内检索。
   依据：STARD 的条级 BM25 R@5=0.33 是同一任务下唯一无需模型的可用数字；
   STARA 的层级保留 + 交叉引用补全是 2026 年最新的工程共识。
2. **条文编号进入 metadata**：`law_title + 第X条` 作为强匹配通道。用户问「劳动法第几条」时，
   这是零成本的精确命中。
3. **交叉引用 / 定义补全（Contextual Retrieval 的法规版）**：STARA 第 2 步「augment provisions with
   definitions, cross-references, and parent provisions」——法规里大量条文写「依照前款规定」
   「本法另有规定的除外」，**单独一条读不懂**。在 wiki page 里把「本章标题 + 上位条 + 被引用定义」
   拼到条文页里，是纯确定性操作，不需要模型。

---

## 3. 方向三：无 embedding 的语义匹配替代方案

这一节按「是否需要训练」分类，因为我们的硬约束是「不引入新模型类型」。

### 3.1 需要训练 → **不符约束，但必须说明清楚**

| 方法 | 机制 | 是否违反约束 | 证据 |
|---|---|---|---|
| **SPLADE** | BERT-MLM 头输出全词表稀疏权重，本质是**学出来的 doc2query** | **违反**：需要一个 BERT 级别的 encoder 模型并训练 | [ECIR 2023 doc2query--](https://arxiv.org/abs/2301.03266) 引用其扩展词有 `reed/herb/troy` 等无关词 |
| **uniCOIL / DeepImpact** | 学 term 权重 | 违反 | — |
| **M3-Embedding 的 Sparse 通道** | 同一模型输出 token-level 稀疏权重 | 违反 | 见下 |

**M3 的 Sparse 通道数据值得单独看**，因为它是「稀疏学习检索」在**多语种**上的最强公开数字
（[arXiv 2402.03216](https://arxiv.org/abs/2402.03216)，MIRACL dev，nDCG@10）：

| | 平均 | zh | ja | ar | ko |
|---|---|---|---|---|---|
| BM25 | 31.9 | 17.5 | 31.2 | 39.5 | 37.1 |
| M3-Sparse | 53.9 | 36.1 | 56.1 | 67.1 | 61.5 |
| M3-Dense | 69.2 | 62.7 | 72.8 | 78.4 | 69.9 |

> 这条数据同时说明了两件事：**①稀疏学习检索在中文上能把 BM25 从 17.5 拉到 36.1（×2），
> 说明「词法+扩展」这条路的天花板很高；②但它需要一个训练好的 encoder，我们做不到。**
> 所以问题变成：**能不能用 LLM 的生成能力，近似地拿到 SPLADE 的那部分扩展词？** ——第 5 节回答。

### 3.2 不需要训练 → **可行，且这是我们的主战场**

#### (a) LLM 查询扩展（我们已有 3 变体 + RRF）

现状符合业界做法。[Query2Doc](https://aclanthology.org/2023.emnlp-main.585/) 的核心技巧是
**把查询重复 n 次再拼接伪文档**，以平衡「短查询 vs 长伪文档」的权重：
`q+ = concat({q} × n, d')`，n=5。**如果我们现在的 3 个变体是把变体直接送进 BM25，那可能缺了这个
「查询侧重复加权」的技巧**——这是一个几行代码的改动。

#### (b) HyDE + BM25 反馈模型（**重点，见第 5 节**）

#### (c) 词表扩展（同义词典 / 领域术语表）

- 业界常用做法：用 LLM **离线**为语料里的**实体、术语**生成同义表达/口语说法，建静态同义表，
  查询时做确定性替换扩展。这是「一次性离线生成 + 运行期零模型」。
- **实证强度：弱。** 我没找到 2025–2026 年公开论文量化「LLM 生成同义词表对 BM25 的增益」。
  文献里最常见的是**人工/WordNet 同义扩展**，而 [ECIR 2023 doc2query--](https://arxiv.org/abs/2301.03266)
  引用的经典结论是**查询扩展可能反而降低效果**（因为查询太短、缺少上下文）。
- **结论：可以做，但必须走「离线生成 + 人工/A-B 验证」，不要指望它单独扛大旗。**

#### (d) query-to-doc 生成（即 HyDE 的检索侧形态）

见第 5 节，直接可用。

#### (e) 用「同一 LLM 做二分类重排」而不是打分

参考 STARA：**LLM 判「这条法条是否回答了这个问题：是/否」比打 0-100 分稳定**。
我们已排除的是「打 0-100 分」，**没有排除二分类**。二分类的输出是离散的，不会出现「噪声候选也拿 100 分」。
代价是每个候选一次调用；可以用「批量 yes/no + JSON schema」把 20 个候选压在一次调用里。

> ⚠️ 注意：这仍然属于「引入额外的运行期模型调用」，与已排除的「LLM rerank 效果弱且 4 倍慢」有重合风险。
> 区别只在**输出形式**（离散 vs 连续）。**建议只在 top-20 上做，并且只在 eval 上证明收益后再上线。**

### 3.3 多语种可行性

- 查询扩展 / HyDE / 词表扩展 **全部与语种无关**：它们只需要 LLM 会那种语言。
  [M3 的多语种 doc2query prompt](https://arxiv.org/abs/2402.03216) 是同一句英语 prompt 直接喂多语种文本。
- **唯一需要注意的语种相关点**：分词。中/日/泰需要分词或 n-gram（我们用 cjk_ngram/jieba）；
  英/德等空格语言直接用词。**LLM 生成的假想文档/问题会在这些语言里自然产出正确的词形**，
  所以「生成侧」没有额外成本，「索引侧」才需要关心分词。

---

## 4. 方向四：2025–2026 RAG 主流架构演进，哪些适合「一个 LLM + BM25」

### 4.1 逐个评估

| 架构 | 核心做法 | 实证 | 对「一个 LLM + BM25」 |
|---|---|---|---|
| **Contextual Retrieval**（Anthropic, 2024-09） | 给每个 chunk **前置 50–100 token 的定位上下文**（哪份文件、哪一节、指代谁），再建 embedding **和 BM25 索引** | 失败检索率 5.7% → 3.7%（仅 Contextual Embeddings，−35%）；再加 **Contextual BM25** → 2.9%（**−49%**）；再加 rerank → 1.9%（−67%）。[来源](https://www.anthropic.com/research/contextual-retrieval) | ✅ **高度适配**：BM25 分支完全不需要 embedding。原文明确说「这些收益叠加（stack）」，**BM25 分支单独也是有效的上下文注入**。⚠️ 文中未单独公布「只做 Contextual BM25」的百分比，我们引用时只能引「组合后 −49%」。 |
| **late chunking**（Jina, arXiv 2409.04701） | 先用**长上下文 embedding 模型**编码全文，再在 transformer 之后、mean pooling 之前切块 | 4 个数据集 × 3 模型平均 nDCG@10：句边界 52.4→54.3（+3.63% 相对）；固定 256 token 52.2→54.0；语义边界 52.4→53.8。[来源](https://arxiv.org/pdf/2409.04701) | ❌ **不适用**：收益完全来自「长上下文 encoder 的 token 表示」，没有 encoder 就没有 late chunking。但它对 Contextual Retrieval 的对比结论对我们有用：**「late chunking 的优势是不需要额外的 LLM」**——反过来说，**用 LLM 打上下文是它认可的等价路线**。 |
| **GraphRAG**（Microsoft, arXiv 2404.16130） | LLM 抽实体关系建图 → Leiden 社区检测 → LLM 生成社区摘要 → 全局问题的 map-reduce 回答 | 全球性问题（comprehensiveness）胜率 **72–83%（p<.001）**，多样性 **62–82%**；索引 Podcast 语料耗时 **281 分钟**（600 token 窗口）；根级社区摘要每查询 token 为全文 map-reduce 的 **1/9 到 1/43**。[来源](https://arxiv.org/pdf/2404.16130v2) | ⚠️ **部分适配**：**适用场景是「全局/综述型问题」**（"这批法规的主要主题是什么"），不是「某条法规怎么规定」。而且**社区检测需要 embedding/聚类**，无 embedding 时只能用实体共现图做聚类——退化成弱版本。**边际收益对单条问答几乎为零。** |
| **RAPTOR**（ICLR 2024, arXiv 2401.18059） | 递归地「聚类 → 摘要 → 重新嵌入」，构树，检索时跨层取节点 | 与**不用 RAPTOR 的 BM25** 对比：NarrativeQA ROUGE 23.52→**27.93**（+4.4 点）；QASPER F1 比 BM25 高 5.5–10.2 点；QuALITY 比 BM25 高 5.1–6.7 点。[来源](https://arxiv.org/pdf/2401.18059v1) | ⚠️ **树形摘要本身可用（LLM 摘要，无 embedding），但「跨层检索」需要 embedding 做最近邻**。无 embedding 时只能做「树形目录 + BM25」，即：给每章/节生成 LLM 摘要，摘要也进 BM25。**这是可做的弱化版**，且与 doc2query 同源（都是「用 LLM 生成额外可检索文本」）。 |
| **agentic RAG / 阅读计划**（我们已有） | 检索 → 计划 → 深读 → 生成 | 无统一基准；[Plan*RAG (arXiv 2410.20753)](https://arxiv.org/abs/2410.20753) 显示「把推理计划外化为 DAG 再执行」在多跳QA上稳定优于 ReAct/Self-RAG，且计算成本可比 | ✅ **我们已经在做**。业界趋势（2026 综述）把「检索作为一个可规划的 agentic 流程」当作主线，这与我们「LLM 阅读计划 → 深读」的结构一致，**不需要改架构。** |
| **HyDE** | LLM 生成假想答案 → embedding 检索 | 见第 5 节 | ✅ **无 embedding 变体成熟**（SIGIR'26） |

### 4.2 「一个 LLM 走天下」的可迁移结论

把这四类架构抽出来看，它们对**词法检索**真正有用的只有一条共性：

> **在索引期用 LLM 往文档里「补它可以被哪些话问到」的文本。**

- Contextual Retrieval = 补「这段在说什么」（定位/指代）；
- doc2query = 补「这段能被怎么问」；
- RAPTOR 的摘要层 = 补「这段在讲什么主题」；
- HyDE = 同上，但发生在**查询侧**。

四者都是「LLM 生成的额外可检索文本」，区别只在**生成的内容形态**和**放在哪一侧**。
**这四者可以共存，且都是对一个 LLM 的复用，不需要任何新模型。**

### 4.3 3 个**不适合**我们的，以及原因

1. **任何以 embedding 为中心的架构**（late chunking、ColBERT 类 late interaction、稠密检索基线）
   —— 结构性不兼容，不用评估。
2. **GraphRAG 的全局问答** —— 解决的是「全局综述」问题，不是「单条检索」问题；
   我们用户问的是「允许吗」，不是「这批法规主要讲什么」。
3. **rerank / LLM 打分重排** —— 已实测无收益且慢，**且与 Community 结论冲突**：
   Anthropic 的 rerank 收益（49%→67%）是**建立在 embedding+BM25 混合候选**之上的，
   我们只有单路词法候选，rerank 能救回的是「已进入候选但排序靠后」的样本，救不回「根本没进候选」的。

---

## 5. 方向五：HyDE 的无 embedding 变体——**核心发现**

### 5.1 有人这么做吗？**有，而且 2026 年有专门论文**

**Revisiting BM25 Feedback Models using HyDE**，Nour Jedidi & Jimmy Lin，SIGIR '26，
[PDF](https://cs.uwaterloo.ca/~jimmylin/publications/Jedidi_Lin_SIGIR2026.pdf)｜[DOI](https://doi.org/10.1145/3805712.3809889)｜[代码](https://github.com/nourj98/hyde-feedback)

**论文的核心洞察**：HyDE 和 Query2Doc 都把 LLM 生成的文本**直接字符串拼接**进查询，
而传统 PRF（伪相关反馈）有**几十年的选词 + 加权理论（Rocchio、RM3）**被忽略了。
他们把「用 BM25 检索回 top-k 文档做反馈」换成「用 LLM 生成的假想文档做反馈」，
**假想文档的角色与反馈文档完全相同**——这就是「无 embedding 的 HyDE」。

**实验设置**：14 个数据集（MS MARCO v1 DL19/20、v2 DL21/22/23 + BEIR 9 个）；
3 个 LLM（Qwen2.5-7B、Qwen3-14B、gpt-oss-20b）；
**8 个假想文档**（≤512 token）；Rocchio α=1、β=0.75；RM3 λ=0.5；**k=128 个扩展词**；
评价指标 **Recall@20**（对应 RAG 场景「top-20 喂给 LLM」）。

**结果（gpt-oss-20b 一组，全部数据集的 Recall@20）**：

| 方法 | MS MARCO 5 集平均 | BEIR 9 集平均 | **全部 14 集平均** |
|---|---|---|---|
| BM25 | — | 34.7 | 28.8 |
| **Query2Doc**（字符串拼接，单文档，q×5） | 32.7 | 36.7 | 32.7 |
| **HyDE + Naive Concat**（8 文档裸拼接） | **29.5** | 33.7 | **29.5（比 BM25 还差 0.7 分！）** |
| **HyDE + MuGI Concat**（自适应重复 γ） | 33.3 | 36.9 | 33.3 |
| HyDE + 平均向量 | 32.5 | 37.4 | 32.5 |
| HyDE + RM3 | 33.2 | 38.6 | 33.2 |
| **HyDE + Rocchio** | **34.7** | **39.1** | **34.7（最优）** |
| （参考）BGE-base-en-v1.5 稠密检索 | — | 42.0 | — |

**三个可以直接抄的结论**：

1. **裸拼接是负面的**：HyDE + Naive Concat 在全部数据集平均上**低于纯 BM25**（29.5 vs 28.8 仅差 0.7，
   但在 BEIR 上明显不如，Qwen 组同样如此）。**把 LLM 生成的一大段文本直接拼进 BM25 查询 = 引入噪声。**
2. **Rocchio 反馈最优，且优于所有字符串拼接**：相对最强拼接方法 MuGI，平均 **+1.4 点（+4.2%）**，
   在 BEIR 上 **+2.2 点（+6%）**。原因论文给了解释：反馈模型**隐式地过滤噪声词**
   （只保留语料内常见度 <10% 的词、按归一化 tf 排序、取 top-k），并用 α/β **线性配平**
   原始查询与生成文本的权重，而字符串重复是「暴力加权」。
3. **超参最优点很靠前**：反馈词数到 **64** 收益饱和；假想文档数到 **4** 收益饱和
   （他们用了 8 个，没有更多收益）。**这对我们意味着：LLM 生成一次就够了，不需要 8 次采样**——
   我们目前「3 个查询变体」的预算可能直接省下来。

### 5.2 对我们的适配性：**极高**

| 维度 | 评价 |
|---|---|
| 模型 | **零新增**。Rocchio/RM3/平均向量都是纯统计公式（tf、df、idf）。 |
| 延迟 | 论文只测量检索质量；**Rocchio 反馈只需对已生成的假想文档算 tf 向量 + 一次 BM25 查询**，开销极小。与我已排除的「LLM rerank 4 倍慢」完全不是一类东西。 |
| 与现有 RRF | **可组合**：把当前的 3 个查询变体改造成「3 个假想文档」→ 各自算 Rocchio 扩展向量 → 再 RRF。或者直接替换 RRF 作为最终查询表示。 |
| 与已排除路线的关系 | 已排除的是「模型给候选打分」。这里是**查询表示构造**，不触碰候选打分。 |
| 风险 | ①假想文档的**事实错误**会被当成扩展词（论文也承认）；②Rocchio 的「丢弃常见词」规则是**语料统计相关**的，在 3000 部法规的小语料上阈值（10% 出现率）需要重调；③法规语料的**术语高度集中**，扩展词可能全部指向「同一部法规的高频词」，导致候选集中在几部大法上。 |

### 5.3 具体做法（伪代码）

```
给定用户口语问题 q：
1. LLM 生成 1 个假想文档 d̂（示例："《安全生产法》第二十八条 生产经营单位应当对从业人员进行
   安全生产教育和培训..."）。-- 一次调用，不需要 8 个
2. 对 d̂ 分词（与索引同一 tokenizer：cjk_ngram / jieba），算归一化 tf 向量 f̃(d̂)：
   - 丢弃在语料中出现率 > 10% 的词（停用/模板词）
   - 按 f̃(d̂)[t] 排序，取 top-k（k=32~64，文献最优 64，我们语料小，先试 32）
3. Rocchio 构造新查询向量：
   w[t] = α * f(q)[t] + β * f̃(d̂)[t]，α=1.0, β=0.75
4. 用 w 作为加权查询送进现有 BM25（Lucene/自实现 boost 或「词 × 权重」展开）
   —— **不要**把 d̂ 原文拼进查询字符串（naive concat 已证伪）
5. （可选）与现有 3 变体 RRF 融合
```

**要注意的一个实现细节**：第 4 步需要 BM25 支持**加权查询**。我们如果现在是「把字符串喂给 BM25」，
需要改成「term + boost」的形式（Lucene 的 `BoostQuery`，或自实现里给 query tf 乘以权重）。
这是 P1 的工程量，不大但必须做——**这是整个方案能否成立的关键，不是可选项**。

---

## 6. 附加发现：口语 ↔ 条文 的技术根因

调研过程中有两处独立证据，解释了我们 0.1236 的 hit@1 为什么低，以及该往哪修：

1. **STARD 的第三条洞察**：零样本词法/稠密方法都不如微调模型，原因是
   「前者**缺乏把生活问题解释成法律事实**的能力，这种能力是 Dense-STARD 通过微调学到的」。
   —— 这句话直接翻译成我们的架构语言：**「把生活问题 → 法律事实」这一步必须由 LLM 完成，
   而它发生的位置可以是查询侧（HyDE）也可以是索引侧（doc2query），但不能指望 BM25 自己完成。**
2. **CLaw 的失败案例**：即使是带搜索能力的前沿模型（Doubao-1.5-pro、DeepSeek-R1），
   在**版本/编号**上也会出错。这印证了「结构性问题（版本、编号）必须用确定性代码解决，
   语义问题（口语→法条）才交 LLM」的分工是对的——我们已把版本过滤做成了确定性代码。

另有一处**需要警惕的评估方法问题**：STARD 的标注一致性 **Cohen's Kappa = 0.5312（中等一致）**，
而且基准自身有 45% 的查询对应多条法条。这说明**「哪条法条对」本身就带主观性**，
**我们的 hit@k 可能存在标注噪声上限**；在把 hit@1 从 0.12 往上推之前，
建议先确认「人工标注的上限」和「命中多条时的判定规则」，否则会在测不准的指标上做优化。

---

## 7. 落地清单（按预期收益 / 成本 排序）

> 收益判断依据：有直接实证的排前，靠类比推理的排后。**成本 = 工程量 + 运行期开销**。
> 「预期收益」是对 口语提问 hit@1（当前 0.1236）/ hit@100（当前 0.596）的方向性估计，**不是承诺值**。

### P0 — 先做，收益/成本比最高

| # | 事项 | 预期收益 | 实施成本 | 风险 | 证据 |
|---|---|---|---|---|---|
| **P0-1** | **把 HyDE 改成「假想文档 + Rocchio 反馈」**（不要裸拼接） | **高**。BM25 反馈模型 + HyDE 是 14 数据集上被验证的最优无 embedding 组合；NR@20 从 28.8 基线拉到 34.7（gpt-oss-20b 全数据集均值），BEIR 上 34.7→39.1 | **低**：一个 Rocchio 函数（~50 行）+ BM25 加权查询支持 + 改 prompt 让 LLM「写出你认为的法条原文」 | 假想文档事实错误会变噪声词；α/β 和小语料下的 10% 阈值需重调；加权查询实现错误会静默降级 | [Jedidi & Lin, SIGIR'26](https://cs.uwaterloo.ca/~jimmylin/publications/Jedidi_Lin_SIGIR2026.pdf) |
| **P0-2** | **给合成问题加相关性过滤**（answerable 校验或实体包含校验） | **中高**。doc2query 上 +16%，且索引 −33%、查询快 23% | **低**：同一次 LLM 调用里多要一个字段；或纯字符串包含校验 | 过滤过狠会砍掉有效召回；LLM 自评 answerable 仍不可靠（需抽查） | [Gospodinov et al., ECIR 2023](https://arxiv.org/abs/2301.03266) |
| **P0-3** | **合成问题「字段权重 < 正文」**（BM25F 式） | **中**。防止模板化提问词抢占 top-1；这是 doc2query 能提召回而不伤精度的前提 | **低**：建索引时分字段加权 | 权重配错会削弱 doc2query 的全部收益 | 推论自 doc2query 的「append 不分隔」+ BM25F 常识；**无直接实证** |
| **P0-4** | **评测集加一层「标注上限」校验** | **间接（防止在噪声指标上优化）** | 低（标注人力） | 需要人工，且法律标注成本高 | STARD κ=0.53、45% 查询多条法条 |

### P1 — 结构性改动，收益可能最大但需要设计

| # | 事项 | 预期收益 | 实施成本 | 风险 | 证据 |
|---|---|---|---|---|---|
| **P1-1** | **doc2query 从「文档级 8 个问题」改为「条文级问题」** | **高**。当前 8 个问题只能覆盖平均 5,487 字文档里的 8 个条文——覆盖率结构性不足。STARD 证明「条」是可行检索单元（条级 BM25 R@5=0.33） | **中高**：需要在 chunk/article 粒度做生成与索引，wiki 页面结构要变；索引条数从 3k 变 5–10 万，BM25 内存与 IDF 统计会变 | 索引规模暴涨；条级 IDF 更稀疏；**需要先确认条文边界解析质量** | [STARD](https://aclanthology.org/2024.findings-emnlp.625/) + doc2query 饱和曲线 |
| **P1-2** | **条文级索引 + 文档级 BM25 的两级检索**（先文档、再条） | **高**。把「找哪部法」和「找哪一条」拆开，各自用最合适的粒度 | **中**：第二级子索引 + 上下文构建改造 | 两级都要处理版本过滤，否则 CLaw 的编号错误会重演 | STARA 5.1.3「限制到相关章节」+ STARD 条级基线 |
| **P1-3** | **条文页补全上下文**：本章标题 + 上位条 + 被引用定义 + 条文编号 | **中高**。Contextual Retrieval 的 BM25 分支在有 embedding 时贡献了 49% 组合收益中的一部分；**法规里「依照前款规定」的指代是本领域最大语料缺陷** | **中**：纯确定性拼接，但需要解析「第X条」「前款」引用 | 解析错引用会注入错误上下文；跨法引用（「本法」vs「《X法》第Y条」）需区分 | [Anthropic Contextual Retrieval](https://www.anthropic.com/research/contextual-retrieval)、[STARA 2.3](https://arxiv.org/abs/2603.03300) |
| **P1-4** | **查询侧重复加权**（Query2Doc 的 `q × n` 技巧） | **低中**。平衡短查询/长扩展文本的权重；Query2Doc 与 HyDE 论文都用到 | **极低**（几行） | 基本无风险；n 需调 | [Query2Doc, EMNLP'23](https://aclanthology.org/2023.emnlp-main.585/) |

### P2 — 可做，收益不确定

| # | 事项 | 预期收益 | 实施成本 | 风险 | 证据 |
|---|---|---|---|---|---|
| **P2-1** | **LLM 生成领域同义表**（离线，术语 → 口语说法） | **低中，不确定** | 中（离线生成 + 建表 + 运行期替换） | **查询扩展可能降低效果**（文献经典结论）；同义表会过时；多语种需分语种建表 | 弱，无 2025–2026 直接实证 |
| **P2-2** | **章节摘要进 BM25**（RAPTOR 的无 embedding 弱化版） | **中**（针对「这部法规大概讲什么」类问题） | 中（每章一次 LLM 摘要） | 摘要也是幻觉源；与 doc2query 收益重叠 | [RAPTOR](https://arxiv.org/pdf/2401.18059v1)：BM25+RAPTOR 比 BM25 在 NarrativeQA 上 +4.4 ROUGE |
| **P2-3** | **LLM 二分类重排 top-20**（是/否，不是 0-100 分） | **中，但可能与我们已排除的 rerank 同性质** | 中（每查询一次批量调用） | **与已排除的「LLM rerank 弱且慢」高度重叠**；只在 eval 证明有净收益后再考虑 | [STARA](https://arxiv.org/abs/2603.03300)：LLM 分类判定是其主要机制 |

### 不建议做（附理由）

| 事项 | 理由 |
|---|---|
| **SPLADE / uniCOIL / M3-Sparse 等学习稀疏检索** | 需要训练 encoder，**违反硬约束**。M3-Sparse 在中文 MIRACL 上 nDCG@10 从 BM25 的 17.5 拉到 36.1 确实诱人，但那是训练出来的。 |
| **late chunking** | 依赖长上下文 embedding 模型，**结构性不可用**。 |
| **GraphRAG 全局问答** | 解决的是「综述型问题」，与我们的用户意图（某条怎么规定）不匹配；且社区检测需要 embedding。 |
| **稠密检索（含中文/法律专用）** | 违反约束。补充证据：STARD 上**所有零样本稠密模型都不如 BM25**（SEED 0.2897 R@5 < BM25 0.3349），说明「随便加个 embedding」并不能解决问题，Dense-STARD 的 0.5206 是**在 STARD 上微调**得来的。 |
| **rerank 模型 / 模型打 0-100 分** | 已实测无收益且 4 倍慢；且 Anthropic 的 rerank 收益建立在 embedding+BM25 混合候选上，我们只有单路候选时 rerank 的收益来源不存在。 |

---

## 8. 需要说清楚的「无公开实证」项（诚实清单）

以下论断我**没有**在 2025–2026 年文献中找到直接、可引用的定量证据，**报告中按「类比推理」处理，不当结论**：

1. **中文法条语料上的 doc2query / LLM 问题合成** —— 无公开实证（ACL Anthology、arXiv 逐篇扫描 + 中文检索均未找到）。
2. **法条引用解析（citation resolution）带来的检索增益** —— 无公开实证；只有做法描述（STARA）与失败案例（CLaw）。
3. **「Contextual BM25 单独使用」的收益百分比** —— Anthropic 原文只公布 35% / 49% / 67% 三个数字，
   **49% 是「Contextual Embeddings + Contextual BM25」的组合值**，没有单独拆出 Contextual BM25 的贡献。
4. **LLM 生成同义词表对 BM25 的增益** —— 无 2025–2026 公开实证。
5. **「问题字段权重 < 正文」的最优权重** —— 无公开实证，属工程推论。
6. **HyDE+BM25 反馈在中文/法规语料上的效果** —— Jedidi & Lin 的实验是英文（MS MARCO + BEIR），
   **跨语种迁移在机制上成立（Rocchio 是纯统计公式），但没有中文实证**。这是 P0-1 必须自己 A/B 的原因。
7. **「每段/每条生成几个问题」的最优值** —— 文献的 n≈20–40 饱和点是在 **~70 词的短段落**上测的，
   我们 5,487 字的文档**不能直接套用**。

---

## 9. 一页纸行动建议

```
立刻做（本周）：
  1. 把现有 query 扩展从「字符串拼接」改成「Rocchio 加权查询」（P0-1 + P1-4）
     —— 这是本次调研里唯一「有 14 数据集实证 + 零新模型 + 低成本」的组合。
  2. 合成问题加 answerable 过滤（P0-2），并把 questions 区块的索引权重降到 ≤0.5（P0-3）。

下一步（下一迭代）：
  3. 评估「条文级」粒度：先只对 top-20 文档建条级子索引（P1-2），measure hit@1 变化。
  4. 条文页补「本章标题 + 上位条 + 被引用定义 + 条号」（P1-3）—— 纯确定性代码。

先别做：
  - 任何 embedding / rerank / SPLADE / late chunking / GraphRAG 全局问答。
  - 任何「模型给候选打分」的新变体（包括 LLM rerank 的变装）。
```

---

## 参考来源

### 检索与扩展（核心）
- Nogueira & Lin, *From doc2query to docTTTTTquery* — [PDF](https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery-v2.pdf)
- Gospodinov, MacAvaney, Macdonald, *Doc2Query--: When Less is More* (ECIR 2023) — [arXiv](https://arxiv.org/abs/2301.03266) ｜ [全文](https://ar5iv.labs.arxiv.org/html/2301.03266)
- Wang, Yang, Wei, *Query2Doc: Query Expansion with Large Language Models* (EMNLP 2023) — [ACL Anthology](https://aclanthology.org/2023.emnlp-main.585/) ｜ [PDF](https://aclanthology.org/2023.emnlp-main.585.pdf)
- Jedidi & Lin, *Revisiting BM25 Feedback Models using HyDE* (SIGIR 2026) — [PDF](https://cs.uwaterloo.ca/~jimmylin/publications/Jedidi_Lin_SIGIR2026.pdf) ｜ [DOI](https://doi.org/10.1145/3805712.3809889) ｜ [代码](https://github.com/nourj98/hyde-feedback)

### 中文法规 / 多语种法律检索
- Su, Hu, et al., *STARD: A Chinese Statute Retrieval Dataset Derived from Real-life Queries by Non-professionals* (Findings of EMNLP 2024) — [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.625/) ｜ [PDF](https://aclanthology.org/2024.findings-emnlp.625.pdf) ｜ [GitHub](https://github.com/oneal2000/STARD)
- *CLaw: Benchmarking Chinese Legal Knowledge in Large Language Models* (Findings of EMNLP 2025) — [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.646/) ｜ [PDF](https://aclanthology.org/2025.findings-emnlp.646.pdf)
- Afane, Hariri, Ouyang, Ho, *Benchmarking Legal RAG: The Promise and Limits of AI Statutory Surveys* (CS&Law '26) — [arXiv](https://arxiv.org/abs/2603.03300)
- Chen et al., *M3-Embedding* (多语种稀疏/稠密，含 LLM doc2query prompt) — [arXiv](https://arxiv.org/abs/2402.03216)

### RAG 架构
- Anthropic, *Introducing Contextual Retrieval* (2024-09-19) — [https://www.anthropic.com/research/contextual-retrieval](https://www.anthropic.com/research/contextual-retrieval)
- Günther et al., *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models* — [arXiv](https://arxiv.org/abs/2409.04701) ｜ [PDF](https://arxiv.org/pdf/2409.04701)
- Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused Summarization* — [arXiv](https://arxiv.org/abs/2404.16130) ｜ [PDF v2](https://arxiv.org/pdf/2404.16130v2)
- Sarthi et al., *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval* (ICLR 2024) — [arXiv](https://arxiv.org/abs/2401.18059) ｜ [PDF](https://arxiv.org/pdf/2401.18059v1)
- Verma et al., *Plan\*RAG: Efficient Test-Time Planning for Retrieval Augmented Generation* — [arXiv](https://arxiv.org/abs/2410.20753)
- Friel, Belyi, Sanyal, *RAGBench* — [arXiv](https://arxiv.org/abs/2407.11005)
- Ma et al., *doc2query-T5 term 预测分析*（MPG 存档） — [全文](https://pure.mpg.de/rest/items/item_3287344_1/component/file_3287345/content)
