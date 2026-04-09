import { useEffect, useMemo } from 'react';
import { useAppStore } from '../stores/appStore';
import { Panel, LoadingButton, StatusLine, Card } from '../components/common';

function formatRate(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function openAdvancedPanel(targetId: string): void {
  const advanced = document.getElementById('advanced-workbench');
  if (advanced instanceof HTMLDetailsElement) {
    advanced.open = true;
  }
  window.requestAnimationFrame(() => {
    document.getElementById(targetId)?.scrollIntoView({
      behavior: 'smooth',
      block: 'start',
    });
  });
}

export function RiskSummaryPanel() {
  const {
    traces,
    tracesLoading,
    tracesError,
    loadTraces,
    loadTrace,
    setSelectedTraceId,
    setTraceConflictOnly,
    setTraceClaimFilter,
  } = useAppStore();

  const openTraceSample = async (
    traceId: string,
    options?: { conflictOnly?: boolean; claimFilter?: 'all' | 'missing_conflict' | 'insufficiency' | 'conditional' },
  ) => {
    setTraceConflictOnly(options?.conflictOnly ?? false);
    setTraceClaimFilter(options?.claimFilter ?? 'all');
    setSelectedTraceId(traceId);
    await loadTrace(traceId);
    openAdvancedPanel('traces-panel');
  };

  useEffect(() => {
    if (!traces.length) {
      loadTraces();
    }
  }, [loadTraces, traces.length]);

  const summary = useMemo(() => {
    const recent = traces.slice(0, 8);
    const total = recent.length;
    const conflicting = recent.filter((trace) => (trace.conflicting_context_count ?? 0) > 0);
    const missingConflictClaim = recent.filter(
      (trace) =>
        (trace.conflicting_context_count ?? 0) > 0 && (trace.conflict_claim_count ?? 0) === 0,
    );
    const insufficiency = recent.filter((trace) => (trace.insufficiency_claim_count ?? 0) > 0);
    const conditional = recent.filter((trace) => (trace.conditional_claim_count ?? 0) > 0);
    const totalContexts = recent.reduce((sum, trace) => sum + (trace.context_count ?? 0), 0);
    const totalConflicts = recent.reduce(
      (sum, trace) => sum + (trace.conflicting_context_count ?? 0),
      0,
    );
    return {
      total,
      conflictingHits: conflicting.length,
      conflictHitRate: total > 0 ? conflicting.length / total : 0,
      contextAvg: total > 0 ? totalContexts / total : 0,
      conflictAvg: total > 0 ? totalConflicts / total : 0,
      missingConflictClaimCount: missingConflictClaim.length,
      insufficiencyCount: insufficiency.length,
      conditionalCount: conditional.length,
      missingConflictSamples: missingConflictClaim.slice(0, 3),
      samples: conflicting.slice(0, 4),
    };
  }, [traces]);

  return (
    <Panel
      title="最近风险态势"
      subtitle="用最近 trace 快速判断系统是否频繁命中冲突知识"
      actions={
        <>
          <LoadingButton loading={tracesLoading} onClick={loadTraces} variant="secondary">
            刷新
          </LoadingButton>
          <button
            type="button"
            className="secondary"
            onClick={() => {
              setTraceConflictOnly(false);
              setTraceClaimFilter('all');
              openAdvancedPanel('traces-panel');
            }}
          >
            查看 Traces
          </button>
        </>
      }
    >
      <StatusLine
        message={
          tracesLoading
            ? '正在加载最近 traces...'
            : tracesError
              ? tracesError
              : undefined
        }
        isError={!!tracesError}
      />
      <div className="metric-grid">
        <div className="metric-card">
          <span>Recent Traces</span>
          <strong>{summary.total}</strong>
        </div>
        <div className="metric-card">
          <span>Conflict Hit Rate</span>
          <strong>{formatRate(summary.conflictHitRate)}</strong>
          <button
            type="button"
            className="secondary"
            onClick={() => {
              setTraceConflictOnly(true);
              setTraceClaimFilter('all');
              openAdvancedPanel('traces-panel');
            }}
            style={{ marginTop: 8 }}
          >
            排查样本
          </button>
        </div>
        <div className="metric-card">
          <span>Conflicting Context Avg</span>
          <strong>{summary.conflictAvg.toFixed(2)}</strong>
        </div>
        <div className="metric-card">
          <span>Missing Conflict Claims</span>
          <strong>{summary.missingConflictClaimCount}</strong>
          <button
            type="button"
            className="secondary"
            onClick={() => {
              setTraceConflictOnly(true);
              setTraceClaimFilter('missing_conflict');
              openAdvancedPanel('traces-panel');
            }}
            style={{ marginTop: 8 }}
          >
            查看缺口
          </button>
        </div>
        <div className="metric-card">
          <span>Insufficiency Traces</span>
          <strong>{summary.insufficiencyCount}</strong>
          <button
            type="button"
            className="secondary"
            onClick={() => {
              setTraceConflictOnly(false);
              setTraceClaimFilter('insufficiency');
              openAdvancedPanel('traces-panel');
            }}
            style={{ marginTop: 8 }}
          >
            查看不足
          </button>
        </div>
        <div className="metric-card">
          <span>Conditional Traces</span>
          <strong>{summary.conditionalCount}</strong>
          <button
            type="button"
            className="secondary"
            onClick={() => {
              setTraceConflictOnly(false);
              setTraceClaimFilter('conditional');
              openAdvancedPanel('traces-panel');
            }}
            style={{ marginTop: 8 }}
          >
            查看条件
          </button>
        </div>
        <div className="metric-card">
          <span>Context Avg</span>
          <strong>{summary.contextAvg.toFixed(2)}</strong>
        </div>
      </div>

      <div style={{ marginTop: 16 }}>
        <div className="section-label">缺少 Conflict Claim 的样本</div>
        <div className="cards">
          {summary.missingConflictSamples.length ? (
            summary.missingConflictSamples.map((trace) => (
              <Card key={`${trace.trace_id}-missing-conflict`} title={trace.query || trace.trace_id}>
                conflicts={trace.conflicting_context_count ?? 0} | conflict_claims=
                {trace.conflict_claim_count ?? 0}
                {'\n'}
                trace_id: {trace.trace_id}
                {'\n'}
                <button
                  type="button"
                  className="secondary"
                  onClick={() =>
                    void openTraceSample(trace.trace_id, {
                      conflictOnly: true,
                      claimFilter: 'missing_conflict',
                    })
                  }
                >
                  打开缺口视图
                </button>
              </Card>
            ))
          ) : (
            <div className="empty-state">最近 traces 里没有“命中冲突但没产出 conflict claim”的样本。</div>
          )}
        </div>
      </div>

      <div style={{ marginTop: 16 }}>
        <div className="section-label">最近冲突样本</div>
        <div className="cards">
          {summary.samples.length ? (
            summary.samples.map((trace) => (
              <Card key={trace.trace_id} title={trace.query || trace.trace_id}>
                conflicts={trace.conflicting_context_count ?? 0} | contexts={trace.context_count ?? 0}
                {'\n'}
                trace_id: {trace.trace_id}
                {'\n'}
                <button
                  type="button"
                  className="secondary"
                  onClick={() =>
                    void openTraceSample(trace.trace_id, {
                      conflictOnly: true,
                      claimFilter: 'all',
                    })
                  }
                >
                  打开 Traces 面板
                </button>
              </Card>
            ))
          ) : (
            <div className="empty-state">最近 traces 里没有命中冲突知识的样本。</div>
          )}
        </div>
      </div>
    </Panel>
  );
}
