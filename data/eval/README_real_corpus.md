# 真实语料回归基线

语料：`data/raw/real/`（4 份中文公文，宁夏移动在岗技术革新奖励活动相关）。

| 文档 | Markdown | 文本块 | 含层级 |
| --- | --- | --- | --- |
| 网络通〔2026〕99号通知 | 2591 字 | 29 | 90% |
| 在岗技术革新系统申报指引 | 1778 字 | 29 | 100% |
| 负面清单及典型案例分析 | 1513 字 | 8 | 88% |
| 在岗技术革新奖励申报模板 | 1394 字 | 25 | 96% |

未含层级的文本块是文档抬头（"急"、发文机关、文号、"附件1"），没有上级章节是正确结果。

回归方式：

```bash
# 解析并摄入
PARSED_OUTPUT_DIR=/tmp/x/parsed WIKI_OUTPUT_DIR=/tmp/x/wiki \
RAG_INGEST_ALLOWED_DIRS=$PWD/data/raw \
python -c "..."   # 见 git log 中的验证脚本

# 检查每个文档的文本块层级覆盖率
python scripts/run_read_plan_eval.py --dataset <read_plan.jsonl>
```

关注点：层级覆盖率不得回退；Markdown 字数不得显著下降（下降说明内容丢失）。
