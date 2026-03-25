import { useState } from 'react';
import { useAppStore } from '../stores/appStore';
import { Panel, StatusLine, LoadingButton, Card, JsonOutput } from '../components/common';

export function ChatPanel() {
  const {
    workspace,
    chatResult,
    chatLoading,
    chatError,
    sendChat,
    feedbackResult,
    feedbackLoading,
    feedbackError,
    submitFeedback,
    selectedTraceId,
    loadTrace,
  } = useAppStore();
  const [query, setQuery] = useState('差旅报销多久内提交？');
  const [topK, setTopK] = useState(6);
  const [feedbackComment, setFeedbackComment] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendChat(query, topK);
  };

  const handleViewTrace = async () => {
    if (selectedTraceId) {
      await loadTrace(selectedTraceId);
    }
  };

  return (
    <Panel
      title="步骤 2 · 提问验证"
      subtitle="输入问题，检查答案、引用、上下文和业务 trace"
    >
      <form onSubmit={handleSubmit}>
        <div className="status-tip">
          当前问题会落在工作区 `{workspace.kbId}` / `{workspace.tenantId || 'default-tenant'}`，
          会话标识为 `{workspace.sessionId || 'session-web'}`。
        </div>
        <label>
          测试问题
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g., 差旅报销多久内提交？"
          />
        </label>
        <div className="two-col">
          <label>
            召回片段数
            <input
              type="number"
              min={1}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value) || 1)}
            />
          </label>
          <div className="actions" style={{ alignItems: 'end' }}>
            <LoadingButton loading={chatLoading} type="submit">
              开始提问
            </LoadingButton>
          </div>
        </div>
        <StatusLine
          message={
            chatLoading
              ? '正在生成回答...'
              : chatError
                ? chatError
                : chatResult?.trace_id
                  ? `回答完成。trace_id=${chatResult.trace_id}`
                  : undefined
          }
          isError={!!chatError}
        />
      </form>

      <div className="stack" style={{ marginTop: 18 }}>
        {chatResult ? (
          <>
            <div className="output answer">{chatResult.answer || '未返回回答'}</div>

            <div>
              <div className="section-label">引用片段</div>
              <div className="cards">
                {chatResult.citations?.length ? (
                  chatResult.citations.map((c, i) => (
                    <Card key={i} title={c.source || c.chunk_id}>
                      chunk_id: {c.chunk_id}
                      {c.score != null ? ` | score: ${c.score}` : ''}
                    </Card>
                  ))
                ) : (
                  <div className="empty-state">当前回答没有返回引用。</div>
                )}
              </div>
            </div>

            <div>
              <div className="section-label">命中上下文</div>
              <div className="cards">
                {chatResult.contexts?.length ? (
                  chatResult.contexts.map((ctx, i) => (
                    <Card
                      key={i}
                      title={
                        (ctx.title as string) ||
                        (ctx.source as string) ||
                        (ctx.chunk_id as string) ||
                        'Context'
                      }
                    >
                      {(ctx.text as string) || JSON.stringify(ctx)}
                    </Card>
                  ))
                ) : (
                  <div className="empty-state">没有可展示的上下文。通常说明还没导入资料或召回为空。</div>
                )}
              </div>
            </div>

            {chatResult.trace_id && (
              <div className="stack">
                <div className="section-label">反馈闭环</div>
                <label>
                  反馈备注（可选）
                  <textarea
                    value={feedbackComment}
                    onChange={(event) => setFeedbackComment(event.target.value)}
                    placeholder="例如：答案正确，但表述偏保守；或者引用不够集中。"
                  />
                </label>
                <div className="actions">
                  <LoadingButton
                    loading={feedbackLoading}
                    type="button"
                    variant="secondary"
                    onClick={() => submitFeedback('up', feedbackComment)}
                  >
                    标记为有帮助
                  </LoadingButton>
                  <LoadingButton
                    loading={feedbackLoading}
                    type="button"
                    variant="secondary"
                    onClick={() => submitFeedback('down', feedbackComment)}
                  >
                    标记为需改进
                  </LoadingButton>
                  <button type="button" className="secondary" onClick={handleViewTrace}>
                    查看本次链路详情
                  </button>
                </div>
                <StatusLine
                  message={
                    feedbackLoading
                      ? '正在提交反馈...'
                      : feedbackError
                        ? feedbackError
                        : feedbackResult
                          ? `反馈已记录。feedback_id=${feedbackResult.feedback_id}`
                          : undefined
                  }
                  isError={!!feedbackError}
                />
                <details className="details-panel">
                  <summary>查看最近问答返回</summary>
                  <JsonOutput data={chatResult} placeholder="还没有问答结果" />
                </details>
              </div>
            )}
          </>
        ) : (
          <div className="empty-state">
            先完成上一步导入，再发起一个问题。这里会显示回答、引用以及送入模型的上下文。
          </div>
        )}
      </div>
    </Panel>
  );
}
