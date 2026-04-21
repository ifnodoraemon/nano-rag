import { useAppStore } from '../stores/appStore';
import { ChatPanel } from './ChatPanel';
import { DocumentsPanel } from './DocumentsPanel';
import { IngestPanel } from './IngestPanel';
import { WorkspacePanel } from './WorkspacePanel';
import { navigateToPage } from '../navigation';

function scrollToSection(sectionId: string): void {
  document.getElementById(sectionId)?.scrollIntoView({
    behavior: 'smooth',
    block: 'start',
  });
}

export function SimpleHomePage() {
  const { health, ingestResult, chatResult } = useAppStore();
  const contexts = Array.isArray(chatResult?.contexts) ? chatResult.contexts : [];
  const conflictingContexts = contexts.filter(
    (context) => context['wiki_status'] === 'conflicting',
  );
  const citationCount = chatResult?.citations?.length ?? 0;
  const activeStep =
    health?.status !== 'ok' ? 1 : !ingestResult ? 2 : !chatResult?.trace_id ? 3 : 4;
  const answerStatus = !chatResult?.answer
    ? '等待回答'
    : conflictingContexts.length > 0
      ? '需要人工复核'
      : citationCount === 0
        ? '依据不足'
        : '回答有依据';
  const answerSummary = !chatResult?.answer
    ? '导入资料后，直接问一个真实问题。'
    : chatResult.answer.length > 180
      ? `${chatResult.answer.slice(0, 180)}...`
      : chatResult.answer;
  const uploadedFiles = ingestResult?.uploaded_files || [];
  const workflowSteps = [
    {
      index: 1,
      label: '环境就绪',
      detail:
        health?.status === 'ok' ? '核心链路已经就绪，可以开始测试。' : '先确认服务和模型网关状态。',
      action: '看状态',
      onClick: () => scrollToSection('simple-journey'),
    },
    {
      index: 2,
      label: '导入资料',
      detail: ingestResult
        ? `最近一次已导入 ${ingestResult.documents} 份资料`
        : '上传这次测试最关键的资料。',
      action: '去导入',
      onClick: () => scrollToSection('simple-ingest-panel'),
    },
    {
      index: 3,
      label: '提问验证',
      detail: chatResult?.trace_id ? '最近一次问答已完成。' : '像真实使用一样提一个具体问题。',
      action: '去提问',
      onClick: () => scrollToSection('chat-panel'),
    },
    {
      index: 4,
      label: '复核结论',
      detail: chatResult?.trace_id ? '查看答案、引用和冲突提示。' : '回答出来后，再判断它是否可信。',
      action: '看结论',
      onClick: () => scrollToSection('simple-review-panel'),
    },
  ];

  return (
    <div className="page-stack">
      <section className="simple-hero">
        <div className="eyebrow">Quick Test</div>
        <h1>上传资料，问一个问题，马上看它答得靠不靠谱。</h1>
        <p>
          这是给普通使用者的默认入口。你不需要先理解 RAG、Trace 或 Benchmark，
          只要上传文件，提问，然后查看答案和依据。
        </p>
        <div className="workflow-strip workflow-strip-compact">
          {workflowSteps.map((step) => {
            const stepState =
              activeStep === step.index ? 'current' : activeStep > step.index ? 'done' : 'pending';
            return (
              <button
                key={step.index}
                type="button"
                className={`workflow-step workflow-step-button ${stepState}`}
                onClick={step.onClick}
              >
                <span>{String(step.index).padStart(2, '0')}</span>
                <strong>{step.label}</strong>
                <p>{step.detail}</p>
                <small>{activeStep === step.index ? `当前步骤 · ${step.action}` : step.action}</small>
              </button>
            );
          })}
        </div>
        <div id="simple-journey" className="journey-grid">
          <div className="journey-card">
            <span>服务状态</span>
            <strong>{health?.status === 'ok' ? '可开始' : '待检查'}</strong>
            <p>{health?.status === 'ok' ? '核心链路已经就绪。' : '建议先确认运行状态面板。'}</p>
          </div>
          <div className="journey-card">
            <span>资料导入</span>
            <strong>{ingestResult ? `${ingestResult.documents} 份已入库` : '尚未导入'}</strong>
            <p>{ingestResult ? `累计 chunks: ${ingestResult.chunks}` : '先上传一批最能代表业务的资料。'}</p>
          </div>
          <div className="journey-card">
            <span>问答复核</span>
            <strong>{chatResult?.trace_id ? '已有最近一次回答' : '还没开始提问'}</strong>
            <p>{chatResult?.trace_id ? '继续核对答案、引用和风险提示。' : '导入完成后立刻问一个真实问题。'}</p>
          </div>
        </div>
      </section>

      <section id="simple-review-panel" className="simple-review">
        <div className="section-label">当前测试摘要</div>
        <div className="simple-review-grid">
          <div className="journey-card">
            <span>测试空间</span>
            <strong>{chatResult?.kb_id || ingestResult?.kb_id || 'default'}</strong>
            <p>
              tenant {chatResult?.tenant_id || ingestResult?.tenant_id || 'n/a'} · session{' '}
              {chatResult?.session_id || '尚未生成'}
            </p>
          </div>
          <div className="journey-card">
            <span>最近一次导入</span>
            <strong>{ingestResult ? `${ingestResult.documents} 份资料` : '尚未导入'}</strong>
            <p>
              {uploadedFiles.length
                ? uploadedFiles.slice(0, 3).join(' / ')
                : '导入后这里会显示本次测试的资料范围。'}
            </p>
          </div>
          <div className="journey-card">
            <span>最近一次回答</span>
            <strong>{answerStatus}</strong>
            <p>
              {chatResult?.trace_id
                ? `引用 ${citationCount} 条，冲突线索 ${conflictingContexts.length} 条`
                : '还没有问答结果。'}
            </p>
          </div>
        </div>
        <div className="simple-review-answer">
          <strong>最近一次回答摘要</strong>
          <p>{answerSummary}</p>
          <div className="actions">
            <button
              type="button"
              className="secondary"
              onClick={() => scrollToSection('chat-panel')}
            >
              查看完整答案与依据
            </button>
            <button
              type="button"
              className="secondary"
              onClick={() => scrollToSection('simple-advanced-settings')}
            >
              调整测试空间
            </button>
            {chatResult?.trace_id ? (
              <button
                type="button"
                className="secondary"
                onClick={() => navigateToPage('investigate', 'traces-panel', 'expert')}
              >
                进入工程排查
              </button>
            ) : null}
          </div>
        </div>
      </section>

      <div id="simple-ingest-panel">
        <IngestPanel audience="simple" />
      </div>
      <DocumentsPanel audience="simple" />
      <div id="chat-panel">
        <ChatPanel audience="simple" />
      </div>

      <details id="simple-advanced-settings" className="details-panel">
        <summary>高级设置：知识库、租户和接口 key</summary>
        <div style={{ marginTop: 12 }}>
          <WorkspacePanel audience="simple" />
        </div>
      </details>
    </div>
  );
}
