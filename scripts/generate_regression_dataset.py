import sys
import json
import asyncio
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import AppContainer
from app.core.tracing import FeedbackStore, TraceStore

def generate_regression_dataset(output_path: Path):
    container = AppContainer.from_env()
    trace_store = container.trace_store
    feedback_store = container.feedback_store
    
    # Retrieve negative feedback
    feedbacks = feedback_store.list(page=1, page_size=1000).items
    negative_traces = []
    for fb in feedbacks:
        if fb.rating == "down":
            trace = trace_store.get(fb.trace_id)
            if trace and trace.query and trace.answer:
                negative_traces.append(trace)
                
    if not negative_traces:
        print("No negative feedback traces found. Dataset generation skipped.")
        return
        
    records = []
    for index, trace in enumerate(negative_traces):
        record = {
            "sample_id": f"regression-fb-{index+1:04d}",
            "kb_id": trace.kb_id or "default",
            "query": trace.query,
            "reference_answer": "EXPECTED: <Determine reference answer from SME>",
            "reference_contexts": [],
            "source_trace_id": trace.trace_id
        }
        records.append(record)
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"Generated regression dataset with {len(records)} records from negative feedback: {output_path}")

if __name__ == "__main__":
    output_file = ROOT / "data/eval/rag_quality_regression.jsonl"
    generate_regression_dataset(output_file)
