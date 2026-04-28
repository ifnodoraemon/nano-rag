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

  const chips: { label: string; status: 'ok' | 'warn' | 'err' | 'neutral' }[] = health
    ? [
        { label: '模型网关', status: health.gateway?.reachable ? 'ok' : 'err' },
        {
          label: 'Phoenix UI',
          status: !health.phoenix?.enabled
            ? 'neutral'
            : health.phoenix?.reachable
              ? 'ok'
              : 'warn',
        },
        {
          label: '向量库',
          status: health.vectorstore?.status === 'ok' ? 'ok' : 'err',
        },
        {
          label: `模型模式: ${health.gateway_mode || 'unknown'}`,
          status: health.gateway_mode === 'live' ? 'ok' : 'warn',
        },
      ] 
    : [];
  const featureChips: { label: string; status: 'ok' | 'neutral' }[] = health
    ? [
        { label: 'Core Chat', status: 'ok' },
        { label: 'Wiki', status: health.features?.wiki ? 'ok' : 'neutral' },
        { label: 'Hybrid', status: health.features?.hybrid_search ? 'ok' : 'neutral' },
        {
          label: 'Diagnosis',
          status: health.features?.diagnosis ? 'ok' : 'neutral',
        },
        { label: 'Eval', status: health.features?.eval ? 'ok' : 'neutral' },
        { label: 'Benchmark', status: health.features?.benchmark ? 'ok' : 'neutral' },
      ]
    : [];
  const authLabel =
    health?.auth_status === 'disabled'
      ? '本地禁用'
      : health?.auth_status === 'configured'
        ? '已配置'
        : '缺少 RAG_API_KEYS';

  return (
    <Panel
      title="运行状态"
      subtitle="确认核心链路是否可测，并识别当前实例开放了哪些扩展能力"
      actions={
        <>
          <LoadingButton loading={healthLoading} onClick={loadHealth} variant="secondary">
            刷新
          </LoadingButton>
          {health?.features?.diagnosis ? (
            <LoadingButton
              loading={diagnosisLoading}
              onClick={() => diagnoseAuto(true)}
              variant="secondary"
            >
              一键 AI 诊断
            </LoadingButton>
          ) : null}
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
                <span>模型网关模式</span>
                <strong>{health.gateway_mode || 'unknown'}</strong>
              </div>
              <div className="info-row">
                <span>向量库模式</span>
                <strong>
                  {health.vectorstore_backend === 'milvus'
                    ? 'Milvus（真实向量库）'
                    : health.vectorstore_backend || 'memory'}
                </strong>
              </div>
              <div className="info-row">
                <span>业务鉴权</span>
                <strong>{authLabel}</strong>
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
            <div>
              <div className="section-label">当前启用能力</div>
              <div className="chip-row">
                {featureChips.map((chip) => (
                  <Chip key={chip.label} label={chip.label} status={chip.status} />
                ))}
              </div>
            </div>
            {health.phoenix?.enabled === false && (
              <div className="status-line">
                Phoenix 未启用。主流程仍可运行，但不会上报外部追踪。
              </div>
            )}
            {!health.features?.diagnosis && (
              <div className="status-line">
                当前实例只开启了 nano core，未启用 diagnosis / eval workbench。
              </div>
            )}
            {health.auth_status === 'missing_keys' && (
              <div className="status-line error">
                API 鉴权已开启但还没有配置 RAG_API_KEYS。请配置业务 key，或仅在本地开发时设置
                RAG_AUTH_DISABLED=true。
              </div>
            )}
            {health.phoenix?.enabled && !health.phoenix?.reachable && (
              <div className="status-line">
                Phoenix UI 当前未连通，不影响问答主流程，但查看追踪时会受影响。
              </div>
            )}
            {!health.gateway?.reachable && health.gateway?.error && (
              <div className="status-line error">
                模型网关不可用：{health.gateway.error}
              </div>
            )}
            <StatusLine
              message={
                health.features?.diagnosis
                  ? diagnosisLoading
                    ? '正在自动诊断最新问题...'
                    : diagnosis?.summary
                  : undefined
              }
              isError={false}
            />
            {health.features?.diagnosis ? (
              <details className="details-panel">
                <summary>查看最近诊断结果</summary>
                <JsonOutput data={diagnosis} placeholder="还没有执行自动诊断" />
              </details>
            ) : null}
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
