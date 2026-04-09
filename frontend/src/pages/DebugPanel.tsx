import { useState } from 'react';
import { useAppStore } from '../stores/appStore';
import { Panel, StatusLine, JsonOutput, LoadingButton, Card } from '../components/common';

type DebugContext = Record<string, unknown>;

export function DebugPanel() {
  const { workspace, debugResult, debugLoading, debugError, runDebug, loadTrace } = useAppStore();
  const [query, setQuery] = useState('差旅报销多久内提交？');
  const [topK, setTopK] = useState(6);
  const contexts = Array.isArray(debugResult?.contexts)
    ? (debugResult.contexts as DebugContext[])
    : [];
  const conflictingContexts = contexts.filter(
    (context) => context['wiki_status'] === 'conflicting',
  );
  const topicContexts = contexts.filter((context) => context['wiki_kind'] === 'topic');
  const conflictingLabels = conflictingContexts
    .map((context) => {
      const title = String(context['title'] || '').trim();
      const chunkId = String(context['chunk_id'] || '').trim();
      return title || chunkId || 'unknown';
    })
    .slice(0, 3);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    runDebug(query, topK);
  };

  const handleViewTrace = async () => {
    if (debugResult?.trace_id) {
      await loadTrace(debugResult.trace_id);
    }
  };

  return (
    <Panel
      title="检索调试"
      subtitle="只查看召回和上下文构建，不执行最终生成"
    >
      <form onSubmit={handleSubmit}>
        <div className="status-tip">
          当前检索调试会使用工作区 `{workspace.kbId}` / `{workspace.tenantId || 'default-tenant'}`，
          方便对照主流程排查召回问题。
        </div>
        <label>
          调试问题
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
              onChange={(e) => setTopK(Number(e.target.value))}
            />
          </label>
          <div className="actions" style={{ alignItems: 'end' }}>
            <LoadingButton loading={debugLoading} type="submit">
              开始调试
            </LoadingButton>
          </div>
        </div>
        <StatusLine
          message={
            debugLoading
              ? '正在执行检索调试...'
              : debugError
                ? debugError
                : debugResult?.trace_id
                  ? `调试完成。trace_id=${debugResult.trace_id}`
                  : undefined
          }
          isError={!!debugError}
        />
        {debugResult && (
          <>
            <div className="cards" style={{ marginBottom: 12 }}>
              <Card title="检索概览">
                contexts={contexts.length} | topics={topicContexts.length}
              </Card>
              <Card title="冲突节点">
                {conflictingContexts.length > 0
                  ? `${conflictingContexts.length} 个，${conflictingLabels.join(' / ')}`
                  : '0 个'}
              </Card>
            </div>

            <div>
              <div className="section-label">命中上下文摘要</div>
              <div className="cards">
                {contexts.length ? (
                  contexts.slice(0, 5).map((context, index) => (
                    <Card
                      key={String(context['chunk_id'] || `debug-${index}`)}
                      title={String(context['title'] || context['chunk_id'] || `Context ${index + 1}`)}
                    >
                      kind={String(context['wiki_kind'] || 'raw')} | status=
                      {String(context['wiki_status'] || 'n/a')}
                      {'\n'}
                      source: {String(context['source'] || 'n/a')}
                    </Card>
                  ))
                ) : (
                  <div className="empty-state">当前没有可展示的上下文摘要。</div>
                )}
              </div>
            </div>
          </>
        )}
        <details className="details-panel">
          <summary>查看原始调试返回</summary>
          <JsonOutput data={debugResult} placeholder="还没有运行检索调试" />
        </details>
        {debugResult?.trace_id && (
          <button
            type="button"
            className="secondary"
            onClick={handleViewTrace}
            style={{ marginTop: 12 }}
          >
            查看链路详情
          </button>
        )}
      </form>
    </Panel>
  );
}
