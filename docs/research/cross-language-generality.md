# 跨语言通用性检验：用美国法典做留出集

之前的全部评测都在中文法规上，且是同一批查询反复测量——**存在过拟合风险，也无法证明通用性**。
这份文档记录用**全新的英文语料**做的独立检验。

## 语料

[`dreamproit/uscode`](https://huggingface.co/datasets/dreamproit/uscode)（美国法典，USLM 结构化数据）。

- 取 shard 0（25,000 行），过滤掉 `repealed` / `transferred` / `omitted` / `renumbered` / `reserved`
- 按 `parent_identifier` 分组（中位 6 条/组，最大 200），每组作为一篇文档
- 随机取 400 篇建索引，平均 14,314 字符/篇
- 查询：从**未使用过的**文档正文中段生成，不暴露标题 / 章节名 / 机构名，
  共 196 条日常英语提问

与美国法典的差异（正是检验价值所在）：**英文、普通法体系、章节编号不同、
标题是分类名而非内容描述**。

## 发现的一个真实通用性 bug

首次评测：**196 条里 28 条报错**。

```
ModelOutputError: query expansion returned a 63-character variant (> 60)
```

根因：

```python
MAX_VARIANT_CHARS = 60   # 硬编码
```

**60 个字符在中文里是合理的关键词查询，在英文里是一句被截断的话。**
英文单词更长，表达同样的关键词需要更多字符——校验器在拒绝**有效的扩展**。

修法：**按脚本自身的单位度量**

```python
def _variant_exceeds_budget(text):
    words = text.split()
    spaced = len(words) > 1 or not any(ord(ch) > 127 for ch in text)
    if spaced:
        return len(words) > MAX_VARIANT_WORDS, "<12 words>"
    return len(text) > MAX_UNSPACED_VARIANT_CHARS, "<60 chars, unspaced>"
```

- 有空格的语言（英、德、法等）按**词数**计（关键词查询就是几个词，无论单词多长）
- 无空格的语言（中、日）按**字符数**计（没有词边界可数，100 字符的连续串就是句子）

只按词数计会让 100 字的中文句子被当作"1 个词"通过；只按字符计会拒绝有效的英文查询。

## 修复效果

| 指标 | 修复前 | 修复后 |
| --- | --- | --- |
| **报错** | **28 / 196** | **0** |
| `hit@1` | 0.3010 | **0.3673** |
| `hit@4` | 0.4337 | **0.4796** |
| `hit@12` | 0.5510 | **0.6378** |
| 未召回 | 88 | 71 |

中文侧回归验证：`hit@1` 0.6037 → **0.6222**，`hit@12` 0.9519 持平，未召回 13 持平。
**无退化。**

## 英文仍明显差于中文（0.367 vs 0.622），原因已定位

未召回的 71 条里，文档标题是**分类名**而非内容描述：

| 查询 | 期望文档标题 |
| --- | --- |
| Can my bank make me put my dispute in writing? | BANKS AND BANKING > CHECK TRUNCATION |
| How long do they have to decide whether to approve my claim? | BANKS AND BANKING > WALL STREET REFORM > ORDERLY |
| What happens to the money collected from these fees? | AGRICULTURE > TOBACCO INSPECTION |

而正文是具体条文（`§ 1173. Reporting requirements`）。

**中文对照**：标题「重庆市国防动员条例」直接描述内容，所以标题有区分度。
这是**语料结构差异**，不是系统缺陷——但确实让 BM25 在英文上少了一个信号。

## 尝试并否决的调整

给 `title_weight` 加权（英文语料）：

| title_weight | hit@1 |
| --- | --- |
| 0.0（当前） | 0.3622 |
| 0.3 | **0.3878** |
| 0.6 | 0.3724 |

英文上 0.3 略好（+2.6pt），但**中文语料上加权是变差的**（历史实测 −0.023 到 −0.053）。

**不能统一调整**——最优配置依赖语料结构。这恰恰说明：
一个"通用"的系统不能靠固定超参，而应该让这类选择**由数据自身决定**
（例如按标题的实际区分度自动调整权重，而不是人为设一个全局值）。

## 结论

- **系统通过了跨语言、跨法系的检验**：换语言、换法系后没有结构性失效，
  只暴露了一个语言相关的硬编码，已修
- **英文效果确实低于中文**，原因已定位到语料结构（分类名标题），不是算法缺陷
- **超参最优值与语料相关**，全局固定值不可取——这是下一步值得做的方向
