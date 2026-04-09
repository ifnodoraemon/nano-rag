import { HealthPanel } from './HealthPanel';
import { QualitySummaryPanel } from './QualitySummaryPanel';
import { RiskSummaryPanel } from './RiskSummaryPanel';
import { WorkspacePanel } from './WorkspacePanel';
import { IngestPanel } from './IngestPanel';
import { ChatPanel } from './ChatPanel';
import { DebugPanel } from './DebugPanel';
import { TracesPanel } from './TracesPanel';
import { EvalPanel } from './EvalPanel';

export function HomePage() {
  return (
    <div className="shell">
      <section className="hero">
        <div className="eyebrow">Nano RAG Validation Workspace</div>
        <h1>先导入知识，再提问验证，必要时再做高级调试。</h1>
        <p>
          这个前端现在突出 NanoRAG 的标准验证闭环：先配置业务工作区，再导入测试知识、验证问答和引用，
          最后把坏例送去诊断和评测。Debug、Trace 和 Eval 仍然保留，但统一收在高级区域，避免主页面像接口控制台。
        </p>
        <div className="workflow-strip">
          <div className="workflow-step">
            <span>01</span>
            绑定 kb_id / tenant_id / session_id 和业务 API Key
          </div>
          <div className="workflow-step">
            <span>02</span>
            导入测试知识，发起问题，验证答案与引用
          </div>
          <div className="workflow-step">
            <span>03</span>
            对坏例做反馈、诊断和 benchmark
          </div>
        </div>
        <div className="hero-links">
          <a className="secondary" href="/docs" target="_blank" rel="noreferrer">
            打开 API 文档
          </a>
        </div>
      </section>

      <div className="main-flow">
        <HealthPanel />
        <RiskSummaryPanel />
        <QualitySummaryPanel />
        <WorkspacePanel />
        <IngestPanel />
        <div id="chat-panel">
          <ChatPanel />
        </div>
      </div>

      <details className="advanced-toggle" id="advanced-workbench">
        <summary>高级调试与评测</summary>
        <div className="advanced-grid">
          <div id="debug-panel">
            <DebugPanel />
          </div>
          <div id="traces-panel">
            <TracesPanel />
          </div>
          <div id="eval-panel">
            <EvalPanel />
          </div>
        </div>
      </details>
    </div>
  );
}
