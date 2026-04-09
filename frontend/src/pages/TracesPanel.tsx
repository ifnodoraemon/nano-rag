import { useEffect, useState } from 'react';
import { useAppStore, type TraceClaimFilter } from '../stores/appStore';
import { Panel, StatusLine, JsonOutput, LoadingButton, Card } from '../components/common';
import { navigateToPage } from '../navigation';

type TraceContext = Record<string, unknown>;
type TraceClaim = {
  claim_type?: string;
  text?: string;
  citation_labels?: string[];
};

function formatMetric(value: number | undefined): string {
  return value === undefined ? 'n/a' : String(value);
}

function matchesTraceClaimFilter(
  trace: {
    conflicting_context_count?: number;
    conflict_claim_count?: number;
    insufficiency_claim_count?: number;
    conditional_claim_count?: number;
  },
  claimFilter: TraceClaimFilter,
): boolean {
  if (claimFilter === 'missing_conflict') {
    return (trace.conflicting_context_count ?? 0) > 0 && (trace.conflict_claim_count ?? 0) === 0;
  }
  if (claimFilter === 'insufficiency') {
    return (trace.insufficiency_claim_count ?? 0) > 0;
  }
  if (claimFilter === 'conditional') {
    return (trace.conditional_claim_count ?? 0) > 0;
  }
  return true;
}

function traceClaimFilterLabel(claimFilter: TraceClaimFilter): string {
  if (claimFilter === 'missing_conflict') {
    return '只看缺少 conflict claim';
  }
  if (claimFilter === 'insufficiency') {
    return '只看 insufficiency claims';
  }
  if (claimFilter === 'conditional') {
    return '只看 conditional claims';
  }
  return '全部 claims';
}

export function TracesPanel() {
  const {
    traces,
    tracesLoading,
    tracesError,
    loadTraces,
    selectedTraceId,
    currentTrace,
    traceLoading,
    traceError,
    loadTrace,
    diagnosis,
    diagnosisLoading,
    diagnosisError,
    diagnoseTrace,
    setSelectedTraceId,
    prepareChatReplay,
    traceConflictOnly,
    traceClaimFilter,
    setTraceConflictOnly,
    setTraceClaimFilter,
  } = useAppStore();

  const [inputTraceId, setInputTraceId] = useState('');
  const contexts = Array.isArray(currentTrace?.contexts)
    ? (currentTrace.contexts as TraceContext[])
    : [];
  const supportingClaims = Array.isArray(currentTrace?.supporting_claims)
    ? (currentTrace.supporting_claims as TraceClaim[])
    : [];
  const conflictingContexts = contexts.filter(
    (context) => context['wiki_status'] === 'conflicting',
  );
  const conflictClaims = supportingClaims.filter(
    (claim) => claim['claim_type'] === 'conflict',
  );
  const insufficiencyClaims = supportingClaims.filter(
    (claim) => claim['claim_type'] === 'insufficiency',
  );
  const conditionalClaims = supportingClaims.filter(
    (claim) => claim['claim_type'] === 'conditional',
  );
  const topicContexts = contexts.filter((context) => context['wiki_kind'] === 'topic');
  const sourceContexts = contexts.filter((context) => context['wiki_kind'] === 'source');
  const diagnosisFindings = diagnosis?.findings || [];
  const visibleTraces = traces.filter((trace) => {
    if (traceConflictOnly && (trace.conflicting_context_count ?? 0) === 0) {
      return false;
    }
    return matchesTraceClaimFilter(trace, traceClaimFilter);
  });

  useEffect(() => {
    loadTraces();
  }, [loadTraces]);

  useEffect(() => {
    if (!selectedTraceId) {
      return;
    }
    setInputTraceId(selectedTraceId);
    if (currentTrace?.trace_id === selectedTraceId) {
      return;
    }
    void loadTrace(selectedTraceId);
  }, [currentTrace?.trace_id, loadTrace, selectedTraceId]);

  const handleLoadTrace = async (e: React.FormEvent) => {
    e.preventDefault();
    if (inputTraceId.trim()) {
      setSelectedTraceId(inputTraceId.trim());
      await loadTrace(inputTraceId.trim());
    }
  };

  const handleTraceClick = async (traceId: string) => {
    setInputTraceId(traceId);
    setSelectedTraceId(traceId);
    await loadTrace(traceId);
  };

  const handleDiagnoseTrace = async (includeAi = false) => {
    if (inputTraceId.trim()) {
      await diagnoseTrace(inputTraceId.trim(), includeAi);
    }
  };

  const handleReplayTrace = () => {
    if (!currentTrace?.query) {
      return;
    }
    const topK =
      typeof currentTrace.retrieval_params?.top_k === 'number'
        ? Number(currentTrace.retrieval_params.top_k)
        : undefined;
    prepareChatReplay({
      query: currentTrace.query,
      kbId: currentTrace.kb_id || undefined,
      tenantId: currentTrace.tenant_id ?? '',
      sessionId: currentTrace.session_id ?? '',
      traceId: currentTrace.trace_id,
      topK,
      sourceLabel: `trace ${currentTrace.trace_id}`,
    });
    navigateToPage('validate');
  };

  return (
    <Panel
      title="最近链路记录"
      subtitle="查看最近问题的 trace 摘要和详情"
      actions={
        <>
          <LoadingButton
            loading={tracesLoading}
            onClick={loadTraces}
            variant="secondary"
          >
            刷新
          </LoadingButton>
          <button
            type="button"
            className="secondary"
            onClick={() => setTraceConflictOnly(!traceConflictOnly)}
          >
            {traceConflictOnly ? '显示全部' : '只看冲突'}
          </button>
        </>
      }
    >
      <details className="details-panel" style={{ marginBottom: 12 }}>
        <summary>筛选与定位</summary>
        <div className="actions" style={{ marginTop: 12 }}>
          <button
            type="button"
            className={`secondary${traceClaimFilter === 'all' ? ' selected-card' : ''}`}
            onClick={() => setTraceClaimFilter('all')}
          >
            全部 claims
          </button>
          <button
            type="button"
            className={`secondary${traceClaimFilter === 'missing_conflict' ? ' selected-card' : ''}`}
            onClick={() => setTraceClaimFilter('missing_conflict')}
          >
            缺少 conflict claim
          </button>
          <button
            type="button"
            className={`secondary${traceClaimFilter === 'insufficiency' ? ' selected-card' : ''}`}
            onClick={() => setTraceClaimFilter('insufficiency')}
          >
            insufficiency
          </button>
          <button
            type="button"
            className={`secondary${traceClaimFilter === 'conditional' ? ' selected-card' : ''}`}
            onClick={() => setTraceClaimFilter('conditional')}
          >
            conditional
          </button>
        </div>
      </details>
      {(traceConflictOnly || traceClaimFilter !== 'all') && (
        <div className="status-line">
          当前筛选:
          {traceConflictOnly ? ' 只看冲突' : ' 全部 traces'} | {traceClaimFilterLabel(traceClaimFilter)}
        </div>
      )}
      <div className="trace-list">
        {tracesError ? (
          <div className="muted">{tracesError}</div>
        ) : visibleTraces.length === 0 ? (
          <div className="muted">还没有 trace 记录</div>
        ) : (
          visibleTraces.slice(0, 8).map((t) => (
            <button
              key={t.trace_id}
              className="trace-item secondary"
              type="button"
              onClick={() => handleTraceClick(t.trace_id)}
            >
              <strong>{t.query || '未命名问题'}</strong>
              <div className="mono">{t.trace_id}</div>
              <div className="muted">
                contexts={t.context_count ?? 0} | conflicts={t.conflicting_context_count ?? 0} | model={t.model_alias || 'n/a'}
              </div>
              <div className="muted">
                conflict_claims={t.conflict_claim_count ?? 0} | insufficiency_claims=
                {t.insufficiency_claim_count ?? 0}
              </div>
              <div className="muted">conditional_claims={t.conditional_claim_count ?? 0}</div>
              {(t.conflicting_context_count ?? 0) > 0 && (
                <div className="muted">命中了冲突知识节点，建议优先查看诊断结果。</div>
              )}
              <div className="muted">
                kb={t.kb_id || 'default'} | tenant={t.tenant_id || 'n/a'} | session={t.session_id || 'n/a'}
              </div>
            </button>
          ))
        )}
      </div>

      <form onSubmit={handleLoadTrace} style={{ marginTop: 16 }}>
        <label>
          Trace ID
          <input
            value={inputTraceId}
            onChange={(e) => setInputTraceId(e.target.value)}
            placeholder="输入 trace_id 查看详情"
          />
        </label>
        <div className="actions">
          <LoadingButton loading={traceLoading} type="submit">
            加载详情
          </LoadingButton>
          <LoadingButton
            loading={diagnosisLoading}
            type="button"
            variant="secondary"
            onClick={() => handleDiagnoseTrace(false)}
          >
            规则诊断
          </LoadingButton>
          <LoadingButton
            loading={diagnosisLoading}
            type="button"
            variant="secondary"
            onClick={() => handleDiagnoseTrace(true)}
          >
            AI 诊断
          </LoadingButton>
          {currentTrace?.query && (
            <button type="button" className="secondary" onClick={handleReplayTrace}>
              回放到 Chat
            </button>
          )}
        </div>
        <StatusLine
          message={
            traceLoading
              ? '正在加载 trace...'
              : traceError
                ? traceError
                : currentTrace
                  ? 'Trace 已加载。'
                  : undefined
          }
          isError={!!traceError}
        />
        {currentTrace && (
          <div className="stack" style={{ marginBottom: 12 }}>
            <div className="metric-grid">
              <div className="metric-card">
                <span>Latency</span>
                <strong>{formatMetric(currentTrace.latency_seconds)}</strong>
              </div>
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
                <strong>{currentTrace.citations?.length ?? 0}</strong>
              </div>
              <div className="metric-card">
                <span>Claims</span>
                <strong>{supportingClaims.length}</strong>
              </div>
              <div className="metric-card">
                <span>Conflict Claims</span>
                <strong>{conflictClaims.length}</strong>
              </div>
              <div className="metric-card">
                <span>Insufficiency Claims</span>
                <strong>{insufficiencyClaims.length}</strong>
              </div>
              <div className="metric-card">
                <span>Conditional Claims</span>
                <strong>{conditionalClaims.length}</strong>
              </div>
            </div>

            <div>
              <div className="section-label">上下文摘要</div>
              <div className="cards">
                {contexts.length ? (
                  contexts.slice(0, 6).map((context, index) => (
                    <Card
                      key={String(context['chunk_id'] || `context-${index}`)}
                      title={String(context['title'] || context['chunk_id'] || `Context ${index + 1}`)}
                    >
                      kind={String(context['wiki_kind'] || 'raw')} | status=
                      {String(context['wiki_status'] || 'n/a')}
                      {'\n'}
                      score={String(context['score'] || 'n/a')}
                      {'\n'}
                      source: {String(context['source'] || context['original_source_path'] || 'n/a')}
                    </Card>
                  ))
                ) : (
                  <div className="empty-state">这个 trace 没有上下文记录。</div>
                )}
              </div>
            </div>

            <div>
              <div className="section-label">Claim 摘要</div>
              <div className="cards">
                {supportingClaims.length ? (
                  supportingClaims.slice(0, 6).map((claim, index) => (
                    <Card
                      key={`${String(claim['text'] || 'claim')}-${index}`}
                      title={`${String(claim['claim_type'] || 'factual')} | ${
                        Array.isArray(claim['citation_labels'])
                          ? (claim['citation_labels'] as string[]).join(', ')
                          : 'n/a'
                      }`}
                    >
                      {String(claim['text'] || 'n/a')}
                    </Card>
                  ))
                ) : (
                  <div className="empty-state">这个 trace 没有结构化 supporting claims。</div>
                )}
              </div>
            </div>

            <div>
              <div className="section-label">冲突节点</div>
              <div className="cards">
                {conflictingContexts.length ? (
                  conflictingContexts.slice(0, 4).map((context, index) => (
                    <Card
                      key={`${String(context['chunk_id'] || 'conflict')}-${index}`}
                      title={String(context['title'] || context['chunk_id'] || `Conflict ${index + 1}`)}
                    >
                      source: {String(context['source'] || context['original_source_path'] || 'n/a')}
                      {'\n'}
                      chunk_id: {String(context['chunk_id'] || 'n/a')}
                    </Card>
                  ))
                ) : (
                  <div className="empty-state">这个 trace 没有命中冲突知识节点。</div>
                )}
              </div>
            </div>
          </div>
        )}
        <StatusLine
          message={
            diagnosisLoading
              ? '正在分析 trace...'
              : diagnosisError
                ? diagnosisError
                : diagnosis
                  ? diagnosis.summary
                  : undefined
          }
          isError={!!diagnosisError}
        />
        {diagnosis && (
          <div className="cards" style={{ marginBottom: 12 }}>
            {diagnosisFindings.length ? (
              diagnosisFindings.slice(0, 4).map((finding, index) => (
                <Card
                  key={`${finding.category}-${index}`}
                  title={`${finding.category} | ${finding.severity}`}
                >
                  {finding.rationale}
                </Card>
              ))
            ) : (
              <div className="empty-state">这次诊断没有返回明确 finding。</div>
            )}
          </div>
        )}
        <details className="details-panel">
          <summary>查看原始 trace 和诊断数据</summary>
          <div className="stack" style={{ marginTop: 12 }}>
            <JsonOutput data={currentTrace} placeholder="还没有加载任何 trace" />
            <JsonOutput data={diagnosis} placeholder="还没有执行诊断" />
          </div>
        </details>
      </form>
    </Panel>
  );
}
