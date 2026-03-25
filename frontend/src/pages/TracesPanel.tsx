import { useEffect, useState } from 'react';
import { useAppStore } from '../stores/appStore';
import { Panel, StatusLine, JsonOutput, LoadingButton } from '../components/common';

export function TracesPanel() {
  const {
    traces,
    tracesLoading,
    tracesError,
    loadTraces,
    currentTrace,
    traceLoading,
    traceError,
    loadTrace,
    diagnosis,
    diagnosisLoading,
    diagnosisError,
    diagnoseTrace,
    setSelectedTraceId,
  } = useAppStore();

  const [inputTraceId, setInputTraceId] = useState('');

  useEffect(() => {
    loadTraces();
  }, [loadTraces]);

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

  return (
    <Panel
      title="最近链路记录"
      subtitle="查看最近问题的 trace 摘要和详情"
      actions={
        <LoadingButton
          loading={tracesLoading}
          onClick={loadTraces}
          variant="secondary"
        >
          刷新
        </LoadingButton>
      }
    >
      <div className="trace-list">
        {tracesError ? (
          <div className="muted">{tracesError}</div>
        ) : traces.length === 0 ? (
          <div className="muted">还没有 trace 记录</div>
        ) : (
          traces.slice(0, 8).map((t) => (
            <button
              key={t.trace_id}
              className="trace-item secondary"
              type="button"
              onClick={() => handleTraceClick(t.trace_id)}
            >
              <strong>{t.query || '未命名问题'}</strong>
              <div className="mono">{t.trace_id}</div>
              <div className="muted">
                contexts={t.context_count ?? 0} | model={t.model_alias || 'n/a'}
              </div>
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
        <JsonOutput data={currentTrace} placeholder="还没有加载任何 trace" />
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
        <JsonOutput data={diagnosis} placeholder="还没有执行诊断" />
      </form>
    </Panel>
  );
}
