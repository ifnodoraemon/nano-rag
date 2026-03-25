import { useEffect } from 'react';
import { useAppStore } from '../stores/appStore';
import { Panel, Chip, JsonOutput, LoadingButton, StatusLine } from '../components/common';

export function HealthPanel() {
  const {
    health,
    healthLoading,
    healthError,
    loadHealth,
    diagnosis,
    diagnosisLoading,
    diagnosisError,
    diagnoseAuto,
  } = useAppStore();

  useEffect(() => {
    loadHealth();
  }, [loadHealth]);

  const chips: { label: string; status: 'ok' | 'warn' | 'err' | 'neutral' }[] = health
    ? [
        { label: '模型网关', status: health.gateway?.reachable ? 'ok' : 'err' },
        { label: 'Phoenix UI', status: health.phoenix?.reachable ? 'ok' : 'err' },
        {
          label: '向量库',
          status: health.vectorstore?.status === 'ok' ? 'ok' : 'err',
        },
        {
          label: `模式: ${health.gateway_mode || 'unknown'}`,
          status: health.gateway_mode === 'live' ? 'ok' : 'warn',
        },
      ]
    : [];

  return (
    <Panel
      title="运行状态"
      subtitle="确认模型、知识库和观测链路是否就绪"
      actions={
        <>
          <LoadingButton loading={healthLoading} onClick={loadHealth} variant="secondary">
            刷新
          </LoadingButton>
          <LoadingButton
            loading={diagnosisLoading}
            onClick={() => diagnoseAuto(true)}
            variant="secondary"
          >
            一键 AI 诊断
          </LoadingButton>
        </>
      }
    >
      <div className="stack">
        <div className="chip-row">
          {chips.map((c) => (
            <Chip key={c.label} label={c.label} status={c.status} />
          ))}
        </div>
        {healthError && <div className="status-line error">{healthError}</div>}
        {diagnosisError && <div className="status-line error">{diagnosisError}</div>}
        {health && (
          <>
            <div className="info-list">
              <div className="info-row">
                <span>服务状态</span>
                <strong>{health.status === 'ok' ? '可测试' : '部分降级'}</strong>
              </div>
              <div className="info-row">
                <span>向量库模式</span>
                <strong>
                  {health.vectorstore_backend === 'milvus'
                    ? 'Milvus（真实向量库）'
                    : health.vectorstore_backend || 'unknown'}
                </strong>
              </div>
              <div className="info-row">
                <span>Trace 数量</span>
                <strong>{health.trace_count}</strong>
              </div>
              <div className="info-row">
                <span>解析产物目录</span>
                <strong className="mono">{health.parsed_dir}</strong>
              </div>
            </div>
            {!health.phoenix?.reachable && (
              <div className="status-line">
                Phoenix UI 当前未连通，不影响问答主流程，但查看追踪时会受影响。
              </div>
            )}
            <StatusLine
              message={diagnosisLoading ? '正在自动诊断最新问题...' : diagnosis?.summary}
              isError={false}
            />
            <details className="details-panel">
              <summary>查看最近诊断结果</summary>
              <JsonOutput data={diagnosis} placeholder="还没有执行自动诊断" />
            </details>
            <details className="details-panel">
              <summary>查看原始健康数据</summary>
              <JsonOutput data={health} placeholder="Loading..." />
            </details>
          </>
        )}
      </div>
    </Panel>
  );
}
