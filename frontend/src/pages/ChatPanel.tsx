import { useEffect, useState } from 'react';
import { useAppStore } from '../stores/appStore';
import { Panel, StatusLine, LoadingButton, Card, JsonOutput } from '../components/common';

type ChatContext = Record<string, unknown>;
type ClaimType = 'factual' | 'conditional' | 'conflict' | 'insufficiency';

export function ChatPanel() {
  const {
    workspace,
    updateWorkspace,
    chatResult,
    chatLoading,
    chatError,
    sendChat,
    feedbackResult,
    feedbackLoading,
    feedbackError,
    submitFeedback,
    diagnosis,
    diagnosisLoading,
    diagnosisError,
    diagnoseTrace,
    selectedTraceId,
    setSelectedTraceId,
    loadTrace,
    chatReplayDraft,
    clearChatReplay,
  } = useAppStore();
  const [query, setQuery] = useState('差旅报销多久内提交？');
  const [topK, setTopK] = useState(6);
  const [feedbackComment, setFeedbackComment] = useState('');
  const [replayNotice, setReplayNotice] = useState<string | null>(null);
  const contexts = Array.isArray(chatResult?.contexts)
    ? (chatResult.contexts as ChatContext[])
    : [];
  const supportingClaims = Array.isArray(chatResult?.supporting_claims)
    ? chatResult.supporting_claims
    : [];
  const claimTypeCounts = supportingClaims.reduce<Record<ClaimType, number>>(
    (acc, claim) => {
      const claimType = (claim.claim_type || 'factual') as ClaimType;
      if (claimType in acc) {
        acc[claimType] += 1;
      }
      return acc;
    },
    { factual: 0, conditional: 0, conflict: 0, insufficiency: 0 },
  );
  const conflictingContexts = contexts.filter(
    (context) => context['wiki_status'] === 'conflicting',
  );
  const topicContexts = contexts.filter((context) => context['wiki_kind'] === 'topic');
  const sourceContexts = contexts.filter((context) => context['wiki_kind'] === 'source');
  const conflictingLabels = conflictingContexts
    .map((context) => {
      const title = String(context['title'] || '').trim();
      const chunkId = String(context['chunk_id'] || '').trim();
      return title || chunkId || 'unknown';
    })
    .slice(0, 3);
  const currentChatTraceId = chatResult?.trace_id || selectedTraceId;
  const chatDiagnosis =
    diagnosis?.target_type === 'trace' && diagnosis.trace_id === currentChatTraceId
      ? diagnosis
      : null;

  useEffect(() => {
    if (!chatReplayDraft) {
      return;
    }
    setQuery(chatReplayDraft.query);
    if (typeof chatReplayDraft.topK === 'number' && chatReplayDraft.topK > 0) {
      setTopK(chatReplayDraft.topK);
    }
    updateWorkspace({
      kbId: chatReplayDraft.kbId || workspace.kbId,
      tenantId:
        chatReplayDraft.tenantId !== undefined ? chatReplayDraft.tenantId : workspace.tenantId,
      sessionId:
        chatReplayDraft.sessionId !== undefined ? chatReplayDraft.sessionId : workspace.sessionId,
    });
    if (chatReplayDraft.traceId) {
      setSelectedTraceId(chatReplayDraft.traceId);
    }
    setReplayNotice(
      `已从 ${chatReplayDraft.sourceLabel || 'trace'} 回填 query 和工作区。确认后重新发起一次在线验证。`,
    );
    clearChatReplay();
    window.requestAnimationFrame(() => {
      document.getElementById('chat-panel')?.scrollIntoView({
        behavior: 'smooth',
        block: 'start',
      });
    });
  }, [
    chatReplayDraft,
    clearChatReplay,
    setSelectedTraceId,
    updateWorkspace,
    workspace.kbId,
    workspace.sessionId,
    workspace.tenantId,
  ]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setReplayNotice(null);
    sendChat(query, topK);
  };

  const handleViewTrace = async () => {
    if (selectedTraceId) {
      await loadTrace(selectedTraceId);
    }
  };

  const handleDiagnoseTrace = async () => {
    if (currentChatTraceId) {
      await diagnoseTrace(currentChatTraceId, false);
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
                : replayNotice
                  ? replayNotice
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
            <div className="metric-grid">
              <div className="metric-card">
                <span>Contexts</span>
                <strong>{contexts.length}</strong>
              </div>
              <div className="metric-card">
                <span>Topic Contexts</span>
                <strong>{topicContexts.length}</strong>
              </div>
              <div className="metric-card">
                <span>Source Contexts</span>
                <strong>{sourceContexts.length}</strong>
              </div>
              <div className="metric-card">
                <span>Conflicting Contexts</span>
                <strong>{conflictingContexts.length}</strong>
              </div>
              <div className="metric-card">
                <span>Citations</span>
                <strong>{chatResult.citations?.length ?? 0}</strong>
              </div>
              <div className="metric-card">
                <span>Claims</span>
                <strong>{supportingClaims.length}</strong>
              </div>
              <div className="metric-card">
                <span>Conflict Claims</span>
                <strong>{claimTypeCounts.conflict}</strong>
              </div>
              <div className="metric-card">
                <span>Insufficiency Claims</span>
                <strong>{claimTypeCounts.insufficiency}</strong>
              </div>
            </div>

            {conflictingContexts.length > 0 && (
              <div className="status-line error" style={{ marginTop: 12 }}>
                本次回答命中了冲突知识节点: {conflictingLabels.join(' / ')}。建议结合引用和 trace
                进一步确认，不要直接当成确定事实。
              </div>
            )}

            <div className="output answer">{chatResult.answer || '未返回回答'}</div>

            <div>
              <div className="section-label">Supporting Claims</div>
              <div className="cards">
                {chatResult.supporting_claims?.length ? (
                  chatResult.supporting_claims.map((claim, i) => (
                    <Card
                      key={`${claim.text}-${i}`}
                      title={
                        `${claim.claim_type || 'factual'} · ${
                          claim.citation_labels?.join(', ') || `Claim ${i + 1}`
                        }`
                      }
                    >
                      {claim.text}
                    </Card>
                  ))
                ) : (
                  <div className="empty-state">当前回答没有返回结构化 supporting claims。</div>
                )}
              </div>
            </div>

            <div>
              <div className="section-label">引用片段</div>
              <div className="cards">
                {chatResult.citations?.length ? (
                  chatResult.citations.map((c, i) => (
                    <Card key={i} title={c.source || c.citation_label || c.chunk_id}>
                      label: {c.citation_label || 'n/a'}
                      {'\n'}
                      chunk_id: {c.chunk_id}
                      {c.score != null ? ` | score: ${c.score}` : ''}
                      {c.evidence_role ? ` | role: ${c.evidence_role}` : ''}
                      {c.wiki_status ? ` | status: ${c.wiki_status}` : ''}
                      {c.span_text ? `\nspan: ${c.span_text}` : ''}
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
                {contexts.length ? (
                  contexts.map((ctx, i) => (
                    <Card
                      key={i}
                      title={
                        (ctx.title as string) ||
                        (ctx.source as string) ||
                        (ctx.chunk_id as string) ||
                        'Context'
                      }
                    >
                      kind={String(ctx.wiki_kind || 'raw')} | status=
                      {String(ctx.wiki_status || 'n/a')}
                      {'\n'}
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
                  <LoadingButton
                    loading={diagnosisLoading}
                    type="button"
                    variant="secondary"
                    onClick={handleDiagnoseTrace}
                  >
                    诊断本次回答
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
                <StatusLine
                  message={
                    diagnosisLoading
                      ? '正在分析本次回答...'
                      : diagnosisError
                        ? diagnosisError
                        : chatDiagnosis
                          ? chatDiagnosis.summary
                          : undefined
                  }
                  isError={!!diagnosisError}
                />
                {chatDiagnosis && (
                  <div className="cards">
                    {chatDiagnosis.findings.length ? (
                      chatDiagnosis.findings.map((finding, index) => (
                        <Card
                          key={`${finding.category}-${index}`}
                          title={`${finding.category} | ${finding.severity}`}
                        >
                          {finding.rationale}
                        </Card>
                      ))
                    ) : (
                      <div className="empty-state">本次诊断没有返回明确 finding。</div>
                    )}
                  </div>
                )}
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
