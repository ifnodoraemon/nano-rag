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
