import { HealthPanel } from './HealthPanel';
import { DebugPanel } from './DebugPanel';
import { TracesPanel } from './TracesPanel';

export function InvestigatePage() {
  return (
    <div className="page-stack">
      <section className="page-intro">
        <div className="eyebrow">Investigate</div>
        <h1>问题答偏了，再来看链路。</h1>
        <p>
          这里专门做排查。先看系统状态，再看检索调试和 trace 详情，不把这些专家工具压在主流程前面。
        </p>
      </section>

      <HealthPanel />
      <DebugPanel />
      <div id="traces-panel">
        <TracesPanel />
      </div>
    </div>
  );
}
