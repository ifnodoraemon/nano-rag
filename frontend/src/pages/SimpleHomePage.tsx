import { ChatPanel } from './ChatPanel';
import { IngestPanel } from './IngestPanel';
import { WorkspacePanel } from './WorkspacePanel';

export function SimpleHomePage() {
  return (
    <div className="page-stack">
      <section className="simple-hero">
        <div className="eyebrow">Quick Test</div>
        <h1>上传资料，问一个问题，马上看它答得靠不靠谱。</h1>
        <p>
          这是给普通使用者的默认入口。你不需要先理解 RAG、Trace 或 Benchmark，
          只要上传文件，提问，然后查看答案和依据。
        </p>
        <div className="simple-steps">
          <div className="simple-step">
            <span>1</span>
            上传一份或多份资料
          </div>
          <div className="simple-step">
            <span>2</span>
            输入你真正想问的问题
          </div>
          <div className="simple-step">
            <span>3</span>
            看答案、依据和风险提示
          </div>
        </div>
      </section>

      <IngestPanel audience="simple" />
      <div id="chat-panel">
        <ChatPanel audience="simple" />
      </div>

      <details className="details-panel">
        <summary>高级设置：知识库、租户和接口 key</summary>
        <div style={{ marginTop: 12 }}>
          <WorkspacePanel audience="simple" />
        </div>
      </details>
    </div>
  );
}
