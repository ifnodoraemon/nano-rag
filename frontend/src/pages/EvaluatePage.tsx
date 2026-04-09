import { EvalPanel } from './EvalPanel';

export function EvaluatePage() {
  return (
    <div className="page-stack">
      <section className="page-intro">
        <div className="eyebrow">Evaluate</div>
        <h1>离线看趋势，线上回放复核。</h1>
        <p>
          这里保留评测、Benchmark 和坏例处理。默认把它和在线问答拆开，避免首次使用时信息过载。
        </p>
      </section>

      <div id="eval-panel">
        <EvalPanel />
      </div>
    </div>
  );
}
