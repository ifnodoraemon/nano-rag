from __future__ import annotations


class RagasRunner:
    def run(self, records: list[dict]) -> dict:
        if not records:
            return {
                "status": "ok",
                "records": 0,
                "aggregate": {
                    "answer_exact_match": 0.0,
                    "reference_context_recall": 0.0,
                    "retrieved_context_count_avg": 0.0,
                },
                "results": [],
            }

        results: list[dict] = []
        exact_matches = 0.0
        context_recalls = 0.0
        retrieved_counts = 0.0

        for record in records:
            answer = str(record.get("answer", "")).strip()
            reference_answer = str(record.get("reference_answer", "")).strip()
            contexts = record.get("retrieved_contexts", []) or []
            reference_contexts = record.get("reference_contexts", []) or []

            exact_match = 1.0 if answer and answer == reference_answer else 0.0
            if reference_contexts:
                matched = 0
                merged_contexts = "\n".join(str(item) for item in contexts)
                for reference_context in reference_contexts:
                    if str(reference_context).strip() and str(reference_context).strip() in merged_contexts:
                        matched += 1
                context_recall = matched / len(reference_contexts)
            else:
                context_recall = 0.0

            results.append(
                {
                    "query": record.get("query"),
                    "answer_exact_match": exact_match,
                    "reference_context_recall": round(context_recall, 4),
                    "retrieved_context_count": len(contexts),
                }
            )
            exact_matches += exact_match
            context_recalls += context_recall
            retrieved_counts += len(contexts)

        total = len(records)
        return {
            "status": "ok",
            "records": total,
            "aggregate": {
                "answer_exact_match": round(exact_matches / total, 4),
                "reference_context_recall": round(context_recalls / total, 4),
                "retrieved_context_count_avg": round(retrieved_counts / total, 4),
            },
            "results": results,
        }
