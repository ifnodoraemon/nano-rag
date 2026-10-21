# Offline Eval Dataset

本目录存放离线评测数据集。当前推荐格式为 JSONL，每行一条记录。

最小必填字段：

- `query`: 测试问题
- `reference_answer`: 金标答案
- `reference_contexts`: 金标证据片段列表

可选字段：

- `top_k`: 该问题评测时的召回数量
- `answer`: 候选答案。缺省时会在评测时调用当前 RAG 系统自动生成
- `retrieved_contexts`: 候选召回上下文。缺省时会在评测时调用当前 RAG 系统自动生成

当前仓库附带一份可直接运行的示例数据集：

- `data/eval/employee_handbook_eval.jsonl`

## 检索阶段评测（discovery recall@k / MRR）

答案级指标（上面的 `reference_answer` / `reference_contexts`）无法回答一个更基础的问题：
**发现阶段有没有把正确文档找出来？** 该阶段是确定性的（wiki BM25 + 版本过滤），
所以可以单独评测、且不需要任何模型调用。

运行：

```bash
python scripts/run_retrieval_eval.py \
  --dataset data/eval/retrieval_employee_handbook.jsonl \
  --min-recall-at 12=0.9 --min-mrr 0.5
```

数据集格式（JSONL，每行一个问题）：

```json
{"sample_id": "r-1", "kb_id": "default", "query": "差旅报销多久内提交？",
 "expected_source_keys": ["policies travel"]}
```

文档身份三选一（可混用，但**建议每行只用一种**——召回率的分母是"期望标识数"）：

- `expected_titles`（推荐用于随仓库分发的数据集）：文档标题，大小写/首尾空格无关。
  对同一份语料最稳定，不受摄入路径影响。
- `expected_source_keys`：版本链身份，由摄入路径派生（`"<末级目录> <标题>"`）。
  路径摄入与上传摄入得到的 key 不同，跨环境不稳定。
- `expected_doc_ids`：精确 doc_id。语料重建后可能变化，适合临时排查。

一个都没给的行会在解析阶段直接报错（不会静默记 0 分）。

注意：召回率统计的是**被覆盖的期望标识数**，不是"命中的文档数"。当语料里存在多份同标题页面时，
若按文档数统计会得出 >1.0 的荒谬结果（这条规则有回归测试覆盖）。

指标：

| 指标 | 含义 |
| --- | --- |
| `recall@k` | top-k 中命中的期望文档占全部期望文档的比例（多期望文档时有意义） |
| `hit@k` | top-k 中是否至少命中一个期望文档 |
| `mrr` | 首个命中文档排名的倒数均值 |
| `candidate_count_avg` | 平均候选文档数（发现层返回给读计划的规模） |
| `unretrieved_records` | 期望文档压根没出现在候选列表里的问题数——这是最值得关注的失败 |

注意：`recall@k` 衡量的是**发现层**。阅读计划（LLM）与深读不在其中，它们由答案级指标覆盖。
