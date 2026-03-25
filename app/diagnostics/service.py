from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.schemas.diagnosis import DiagnosisFinding, DiagnosisResponse
from app.schemas.trace import TraceRecord


def _normalize_text(raw: object) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    text = re.sub(r"(\*\*|__|`)", "", text)
    text = re.sub(r"\s*\[[^\[\]]+:[^\[\]]+\](?=[。！？.!?]?\s*$)", "", text)
    text = re.sub(r"[。！？.!?]+\s*$", "", text)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def _looks_like_refusal(answer: str) -> bool:
    refusal_markers = (
        "无法确认",
        "信息不足",
        "证据不足",
        "没有可用上下文",
        "请提供",
    )
    return any(marker in answer for marker in refusal_markers)


@dataclass
class DiagnosisService:
    generation_client: object | None = None

    def diagnose_trace(self, trace: TraceRecord) -> DiagnosisResponse:
        findings: list[DiagnosisFinding] = []
        contexts = trace.contexts or []
        retrieved = trace.retrieved or []
        answer = str(trace.answer or "")

        if not retrieved:
            findings.append(
                DiagnosisFinding(
                    category="retrieval_empty",
                    severity="high",
                    rationale="检索阶段没有返回任何候选 chunk，问题大概率在 ingest、embedding 或向量检索参数。",
                    suggested_actions=[
                        "确认文档已成功 ingest 且向量库里有数据。",
                        "检查 embedding 模型和 collection 维度是否一致。",
                        "先放大 retrieval top_k，再观察 retrieve/debug 输出。",
                    ],
                    evidence={"retrieved_count": 0, "context_count": len(contexts)},
                )
            )

        elif contexts and not answer:
            findings.append(
                DiagnosisFinding(
                    category="generation_empty",
                    severity="high",
                    rationale="已经拿到了上下文，但生成阶段没有返回答案。",
                    suggested_actions=[
                        "检查 generation provider 返回体和 finish_reason。",
                        "确认 prompt 构造和 answer formatter 没有把内容清空。",
                    ],
                    evidence={
                        "context_count": len(contexts),
                        "generation_finish_reason": trace.generation_finish_reason,
                    },
                )
            )

        elif contexts and _looks_like_refusal(answer):
            findings.append(
                DiagnosisFinding(
                    category="generation_refusal_with_context",
                    severity="medium",
                    rationale="上下文已经命中，但模型仍然给出了拒答/信息不足类型回答，问题更像 prompt 约束或模型稳定性。",
                    suggested_actions=[
                        "收紧 system prompt，明确命中证据后必须优先直接回答。",
                        "在坏例上对比不同 generation 模型或温度设置。",
                        "把命中的关键句前置到 prompt 中，降低模型保守拒答概率。",
                    ],
                    evidence={
                        "context_count": len(contexts),
                        "answer_preview": answer[:200],
                        "finish_reason": trace.generation_finish_reason,
                    },
                )
            )

        if retrieved and trace.reranked_chunk_ids and trace.retrieved_chunk_ids:
            if set(trace.reranked_chunk_ids) - set(trace.retrieved_chunk_ids):
                findings.append(
                    DiagnosisFinding(
                        category="rerank_inconsistent",
                        severity="high",
                        rationale="rerank 结果包含未出现在 retrieval 的 chunk_id，说明链路记录存在不一致。",
                        suggested_actions=[
                            "检查 rerank 返回索引与原 hits 的映射关系。",
                            "确认 trace 记录没有被后续请求覆盖。",
                        ],
                        evidence={
                            "retrieved_chunk_ids": trace.retrieved_chunk_ids,
                            "reranked_chunk_ids": trace.reranked_chunk_ids,
                        },
                    )
                )

        if contexts and retrieved and len(contexts) < min(len(retrieved), 1):
            findings.append(
                DiagnosisFinding(
                    category="context_trimmed",
                    severity="low",
                    rationale="retrieval 有结果，但最终上下文数量较少，可能是 final_contexts 限制导致信息压缩。",
                    suggested_actions=[
                        "检查 retrieval.final_contexts 配置是否过小。",
                    ],
                    evidence={
                        "retrieved_count": len(retrieved),
                        "context_count": len(contexts),
                        "retrieval_params": trace.retrieval_params,
                    },
                )
            )

        if not findings:
            findings.append(
                DiagnosisFinding(
                    category="no_obvious_issue",
                    severity="info",
                    rationale="从当前 trace 看不出明显结构性问题，若答案仍不稳定，优先结合 eval bad cases 做批量对比。",
                    suggested_actions=[
                        "查看同一问题在评测集里的历史表现。",
                        "结合 prompt 版本和模型版本做 A/B 对比。",
                    ],
                    evidence={
                        "retrieved_count": len(retrieved),
                        "context_count": len(contexts),
                        "finish_reason": trace.generation_finish_reason,
                    },
                )
            )

        summary = "；".join(finding.rationale for finding in findings[:2])
        return DiagnosisResponse(
            target_type="trace",
            trace_id=trace.trace_id,
            sample_id=trace.sample_id,
            summary=summary,
            findings=findings,
        )

    def diagnose_eval_result(self, report: dict, result_index: int) -> DiagnosisResponse:
        results = report.get("results", []) if isinstance(report, dict) else []
        if result_index < 0 or result_index >= len(results):
            raise IndexError(f"eval result index out of range: {result_index}")
        result = results[result_index]
        findings: list[DiagnosisFinding] = []

        exact_match = float(result.get("answer_exact_match", 0.0) or 0.0)
        context_recall = float(result.get("reference_context_recall", 0.0) or 0.0)
        answer = str(result.get("answer", "") or "")
        reference_answer = str(result.get("reference_answer", "") or "")

        if context_recall < 1.0:
            findings.append(
                DiagnosisFinding(
                    category="retrieval_gap",
                    severity="high",
                    rationale="参考证据没有完整出现在 retrieved_contexts 中，问题优先在检索层。",
                    suggested_actions=[
                        "检查 chunk 切分是否过粗或过细。",
                        "放大 retrieval top_k 并观察正确 chunk 排名。",
                        "必要时引入 rerank 或更换 embedding 模型。",
                    ],
                    evidence={
                        "reference_context_recall": context_recall,
                        "answer_exact_match": exact_match,
                    },
                )
            )

        if context_recall >= 1.0 and exact_match < 1.0:
            findings.append(
                DiagnosisFinding(
                    category="generation_misalignment",
                    severity="medium",
                    rationale="检索证据已经完整命中，但最终答案仍未命中参考答案，问题优先在生成层或评测口径。",
                    suggested_actions=[
                        "检查 system prompt 是否过度保守或过于啰嗦。",
                        "人工检查 exact_match 是否被格式差异放大。",
                        "考虑增加语义相似度或 LLM judge 指标，而不只看 exact match。",
                    ],
                    evidence={
                        "reference_context_recall": context_recall,
                        "answer_exact_match": exact_match,
                        "answer_preview": answer[:200],
                        "reference_preview": reference_answer[:200],
                    },
                )
            )

        if _looks_like_refusal(answer) and context_recall >= 1.0:
            findings.append(
                DiagnosisFinding(
                    category="refusal_bad_case",
                    severity="medium",
                    rationale="坏例属于命中证据后仍拒答，通常是 prompt 约束或模型不稳定。",
                    suggested_actions=[
                        "把这条样本加入回归集，专门验证拒答问题。",
                        "强化 prompt 中“命中证据后直接回答”的约束。",
                    ],
                    evidence={"answer_preview": answer[:200]},
                )
            )

        if not findings:
            findings.append(
                DiagnosisFinding(
                    category="no_obvious_issue",
                    severity="info",
                    rationale="这条评测样本没有暴露明显结构性问题。",
                    suggested_actions=["继续查看坏例聚类，而不是单条样本。"],
                    evidence={
                        "reference_context_recall": context_recall,
                        "answer_exact_match": exact_match,
                    },
                )
            )

        summary = "；".join(finding.rationale for finding in findings[:2])
        return DiagnosisResponse(
            target_type="eval_result",
            trace_id=result.get("trace_id"),
            sample_id=result.get("sample_id"),
            summary=summary,
            findings=findings,
        )

    async def add_ai_suggestion(self, diagnosis: DiagnosisResponse, payload: dict) -> DiagnosisResponse:
        if self.generation_client is None:
            return diagnosis

        messages = [
            {
                "role": "system",
                "content": (
                    "你是 RAG 系统诊断助手。请根据给定的 trace/eval 诊断结果，"
                    "输出 3 条高优先级改进建议，要求短、具体、可执行。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "diagnosis": diagnosis.model_dump(),
                        "payload": payload,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            result = await self.generation_client.generate(messages)
            diagnosis.ai_suggestion = str(result.get("content", "")).strip() or None
        except Exception as exc:  # pragma: no cover
            diagnosis.ai_suggestion = f"AI diagnosis unavailable: {exc}"
        return diagnosis
