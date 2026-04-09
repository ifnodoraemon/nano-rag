import { WorkspacePanel } from './WorkspacePanel';
import { IngestPanel } from './IngestPanel';
import { ChatPanel } from './ChatPanel';

export function ValidatePage() {
  return (
    <div className="page-stack">
      <section className="page-intro">
        <div className="eyebrow">Validate</div>
        <h1>先导入，再提问，再看引用。</h1>
        <p>
          这里只保留最短验证路径。先确认工作区，再导入资料，然后直接问一个问题看答案和引用是否可信。
        </p>
      </section>

      <WorkspacePanel audience="expert" />
      <IngestPanel audience="expert" />
      <div id="chat-panel">
        <ChatPanel audience="expert" />
      </div>
    </div>
  );
}
