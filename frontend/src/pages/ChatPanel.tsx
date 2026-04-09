import { useEffect, useMemo, useState } from 'react';
import { useAppStore } from '../stores/appStore';
import { Panel, StatusLine, LoadingButton, Card, JsonOutput } from '../components/common';
import { navigateToPage } from '../navigation';

type ChatContext = Record<string, unknown>;
type ClaimType = 'factual' | 'conditional' | 'conflict' | 'insufficiency';

interface ChatPanelProps {
  audience?: 'simple' | 'expert';
}

function buildAnswerStatus(args: {
  hasAnswer: boolean;
  citationCount: number;
  conflictingCount: number;
}): { title: string; detail: string } {
  if (!args.hasAnswer) {
    return { title: '等待回答', detail: '先发起一次提问。' };
  }
  if (args.conflictingCount > 0) {
    return { title: '需要人工再确认', detail: '资料里可能有说法不一致。' };
  }
  if (args.citationCount === 0) {
    return { title: '依据不足', detail: '回答没有带出足够依据。' };
  }
  return { title: '回答有依据', detail: '当前回答带有可追溯引用。' };
}

export function ChatPanel({ audience = 'expert' }: ChatPanelProps) {
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
  const answerStatus = useMemo(
    () =>
      buildAnswerStatus({
        hasAnswer: Boolean(chatResult?.answer),
        citationCount: chatResult?.citations?.length ?? 0,
        conflictingCount: conflictingContexts.length,
      }),
    [chatResult?.answer, chatResult?.citations?.length, conflictingContexts.length],
  );

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
      `已从 ${chatReplayDraft.sourceLabel || 'trace'} 回填问题。确认后重新发起一次测试。`,
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
    sendChat(query, audience === 'simple' ? undefined : topK);
  };

  const handleViewTrace = async () => {
    const traceId = currentChatTraceId || selectedTraceId;
    if (!traceId) {
      return;
    }
    await loadTrace(traceId);
    navigateToPage('investigate', 'traces-panel');
  };

  const handleDiagnoseTrace = async () => {
    if (currentChatTraceId) {
      await diagnoseTrace(currentChatTraceId, false);
    }
  };

  return (
    <Panel
      title={audience === 'simple' ? '步骤 2 · 提问测试' : '提问验证'}
      subtitle={
        audience === 'simple'
          ? '像真实使用一样问一个问题，直接看答案和依据'
          : '默认先看答案和引用，需要时再展开证据和诊断'
      }
    >
      <form onSubmit={handleSubmit}>
        <div className="status-tip">
          {audience === 'simple'
            ? '尽量提一个明确的问题，比如“差旅报销要在多久内提交？”而不是“总结一下制度”。'
            : `当前提问会使用工作区 \`${workspace.kbId}\` / \`${workspace.tenantId || 'default-tenant'}\`，会话标识是 \`${workspace.sessionId || 'session-web'}\`。`}
        </div>
        <label>
          你的问题
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="例如：差旅报销需要在多久内提交？"
          />
        </label>

        {audience === 'expert' ? (
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
        ) : (
          <div className="actions">
            <LoadingButton loading={chatLoading} type="submit">
              开始测试
            </LoadingButton>
          </div>
        )}

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
            <div className="metric-grid compact-metric-grid">
              <div className="metric-card">
                <span>结果判断</span>
                <strong>{answerStatus.title}</strong>
              </div>
              <div className="metric-card">
                <span>说明</span>
                <strong>{answerStatus.detail}</strong>
              </div>
              <div className="metric-card">
                <span>引用</span>
                <strong>{chatResult.citations?.length ?? 0}</strong>
              </div>
              <div className="metric-card">
                <span>风险</span>
                <strong>{conflictingContexts.length > 0 ? '有冲突' : '未发现冲突'}</strong>
              </div>
            </div>

            {conflictingContexts.length > 0 ? (
              <div className="status-line error" style={{ marginTop: 12 }}>
                当前资料里可能存在说法不一致：{conflictingLabels.join(' / ')}。
              </div>
            ) : null}

            <div className="output answer">{chatResult.answer || '未返回回答'}</div>

            <div>
              <div className="section-label">答案依据</div>
              <div className="cards">
                {chatResult.citations?.length ? (
                  chatResult.citations.map((citation, index) => (
                    <Card
                      key={`${citation.chunk_id}-${index}`}
                      title={`${citation.citation_label || 'n/a'} · ${citation.evidence_role || 'evidence'}`}
                    >
                      source: {citation.source || 'n/a'}
                      {citation.span_text ? `\n依据片段: ${citation.span_text}` : ''}
                    </Card>
                  ))
                ) : (
                  <div className="empty-state">当前回答没有返回引用依据。</div>
                )}
              </div>
            </div>

            <div className="actions">
              <LoadingButton
                loading={feedbackLoading}
                type="button"
                variant="secondary"
                onClick={() => submitFeedback('up', feedbackComment)}
              >
                结果靠谱
              </LoadingButton>
              <LoadingButton
                loading={feedbackLoading}
                type="button"
                variant="secondary"
                onClick={() => submitFeedback('down', feedbackComment)}
              >
                结果不靠谱
              </LoadingButton>
              <button type="button" className="secondary" onClick={handleViewTrace}>
                查看处理过程
              </button>
              {audience === 'expert' ? (
                <LoadingButton
                  loading={diagnosisLoading}
                  type="button"
                  variant="secondary"
                  onClick={handleDiagnoseTrace}
                >
                  规则诊断
                </LoadingButton>
              ) : null}
            </div>

            <label>
              备注（可选）
              <textarea
                value={feedbackComment}
                onChange={(event) => setFeedbackComment(event.target.value)}
                placeholder="例如：答案正确，但引用不够集中。"
              />
            </label>

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
              <summary>{audience === 'simple' ? '查看更多细节' : '查看 claims、证据和完整返回'}</summary>
              <div className="stack" style={{ marginTop: 12 }}>
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
                    <span>Claims</span>
                    <strong>{supportingClaims.length}</strong>
                  </div>
                </div>

                <div className="metric-grid">
                  <div className="metric-card">
                    <span>Conflict Claims</span>
                    <strong>{claimTypeCounts.conflict}</strong>
                  </div>
                  <div className="metric-card">
                    <span>Insufficiency Claims</span>
                    <strong>{claimTypeCounts.insufficiency}</strong>
                  </div>
                  <div className="metric-card">
                    <span>Conditional Claims</span>
                    <strong>{claimTypeCounts.conditional}</strong>
                  </div>
                  <div className="metric-card">
                    <span>Factual Claims</span>
                    <strong>{claimTypeCounts.factual}</strong>
                  </div>
                </div>

                <div>
                  <div className="section-label">Supporting Claims</div>
                  <div className="cards">
                    {chatResult.supporting_claims?.length ? (
                      chatResult.supporting_claims.map((claim, index) => (
                        <Card
                          key={`${claim.text}-${index}`}
                          title={`${claim.claim_type || 'factual'} · ${claim.citation_labels?.join(', ') || `Claim ${index + 1}`}`}
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
                  <div className="section-label">命中上下文</div>
                  <div className="cards">
                    {contexts.length ? (
                      contexts.map((context, index) => (
                        <Card
                          key={String(context['chunk_id'] || `context-${index}`)}
                          title={
                            (context['title'] as string) ||
                            (context['source'] as string) ||
                            (context['chunk_id'] as string) ||
                            'Context'
                          }
                        >
                          kind={String(context['wiki_kind'] || 'raw')} | status=
                          {String(context['wiki_status'] || 'n/a')}
                          {'\n'}
                          {(context['text'] as string) || JSON.stringify(context)}
                        </Card>
                      ))
                    ) : (
                      <div className="empty-state">没有可展示的上下文。</div>
                    )}
                  </div>
                </div>

                {chatDiagnosis || audience === 'expert' ? (
                  <>
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
                    {chatDiagnosis ? (
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
                    ) : null}
                  </>
                ) : null}

                <JsonOutput data={chatResult} placeholder="还没有问答结果" />
              </div>
            </details>
          </>
        ) : (
          <div className="empty-state">
            {audience === 'simple'
              ? '先上传一份资料，然后问一个你真正关心的问题。'
              : '先完成导入，再发起一个具体问题。这里默认只展示答案和引用。'}
          </div>
        )}
      </div>
    </Panel>
  );
}
