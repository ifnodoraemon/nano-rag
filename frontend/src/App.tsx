import { useEffect, useState } from 'react';
import { Chip } from './components/common';
import { getPageFromHash, navigateToPage, type AppPage } from './navigation';
import { SimpleHomePage } from './pages/SimpleHomePage';
import { ValidatePage } from './pages/ValidatePage';
import { InvestigatePage } from './pages/InvestigatePage';
import { EvaluatePage } from './pages/EvaluatePage';
import { useAppStore } from './stores/appStore';
import './index.css';

type AudienceMode = 'simple' | 'expert';

const PAGE_META: Record<
  AppPage,
  { title: string; description: string }
> = {
  validate: {
    title: '验证',
    description: '最短路径：配置工作区，导入资料，提问验证。',
  },
  investigate: {
    title: '排查',
    description: '只在答案异常时进入这里，看 health、debug 和 trace。',
  },
  evaluate: {
    title: '评测',
    description: '跑离线评测、查看坏例，再回放到在线问答。',
  },
};

function App() {
  const [page, setPage] = useState<AppPage>(() => getPageFromHash(window.location.hash));
  const [mode, setMode] = useState<AudienceMode>('simple');
  const { workspace, health, healthLoading, loadHealth } = useAppStore();

  useEffect(() => {
    void loadHealth();
  }, [loadHealth]);

  useEffect(() => {
    const onHashChange = () => {
      setPage(getPageFromHash(window.location.hash));
    };
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const pageMeta = PAGE_META[page];
  const gatewayStatus = healthLoading
    ? 'neutral'
    : health?.gateway?.reachable
      ? 'ok'
      : 'err';
  const vectorStatus = healthLoading
    ? 'neutral'
    : health?.vectorstore?.status === 'ok'
      ? 'ok'
      : 'err';
  const serviceStatus = healthLoading
    ? 'neutral'
    : health?.status === 'ok'
      ? 'ok'
      : 'warn';

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand-block">
          <div className="eyebrow">NanoRAG Workspace</div>
          <div className="brand-row">
            <h1>NanoRAG</h1>
            <p>
              {mode === 'simple'
                ? '默认先帮普通用户把测试跑起来。'
                : '保留工程排查和评测能力，但不再强迫所有人先学内部概念。'}
            </p>
          </div>
        </div>

        <div className="mode-switch" role="tablist" aria-label="audience mode">
          {(['simple', 'expert'] as AudienceMode[]).map((targetMode) => (
            <button
              key={targetMode}
              type="button"
              className={mode === targetMode ? 'primary mode-tab active' : 'secondary mode-tab'}
              onClick={() => setMode(targetMode)}
            >
              <span>{targetMode === 'simple' ? '普通视角' : '工程视角'}</span>
              <small>
                {targetMode === 'simple'
                  ? '快速测试：上传文件、提问、看答案'
                  : '专家模式：看链路、排查问题、跑评测'}
              </small>
            </button>
          ))}
        </div>

        {mode === 'expert' ? (
          <div className="nav-tabs" role="tablist" aria-label="workspace sections">
            {(['validate', 'investigate', 'evaluate'] as AppPage[]).map((targetPage) => (
              <button
                key={targetPage}
                type="button"
                className={page === targetPage ? 'primary nav-tab active' : 'secondary nav-tab'}
                onClick={() => navigateToPage(targetPage)}
              >
                <span>{PAGE_META[targetPage].title}</span>
                <small>{PAGE_META[targetPage].description}</small>
              </button>
            ))}
          </div>
        ) : null}

        {mode === 'simple' ? (
          <div className="summary-strip">
            <div className="summary-card">
              <span>当前模式</span>
              <strong>快速测试</strong>
              <p>默认只保留上传资料、提问和结果判断。</p>
            </div>
            <div className="summary-card">
              <span>支持资料</span>
              <strong>PDF / Markdown / TXT / HTML</strong>
              <p>可以一次上传多份资料，系统会把它们当成这次测试的依据。</p>
            </div>
            <div className="summary-card summary-health">
              <span>运行状态</span>
              <div className="chip-row">
                <Chip label="Service" status={serviceStatus} />
                <Chip label="Gateway" status={gatewayStatus} />
                <Chip label="Vector" status={vectorStatus} />
              </div>
              <p>{health?.gateway_mode ? `mode: ${health.gateway_mode}` : '等待健康状态'}</p>
            </div>
          </div>
        ) : (
          <div className="summary-strip">
            <div className="summary-card">
              <span>当前页</span>
              <strong>{pageMeta.title}</strong>
              <p>{pageMeta.description}</p>
            </div>
            <div className="summary-card">
              <span>工作区</span>
              <strong>{workspace.kbId}</strong>
              <p>
                tenant {workspace.tenantId || 'n/a'} · session {workspace.sessionId || 'n/a'}
              </p>
            </div>
            <div className="summary-card summary-health">
              <span>运行状态</span>
              <div className="chip-row">
                <Chip label="Service" status={serviceStatus} />
                <Chip label="Gateway" status={gatewayStatus} />
                <Chip label="Vector" status={vectorStatus} />
              </div>
              <p>{health?.gateway_mode ? `mode: ${health.gateway_mode}` : '等待健康状态'}</p>
            </div>
          </div>
        )}
      </header>

      <main className="page-shell">
        {mode === 'simple' ? <SimpleHomePage /> : null}
        {mode === 'expert' && page === 'validate' ? <ValidatePage /> : null}
        {mode === 'expert' && page === 'investigate' ? <InvestigatePage /> : null}
        {mode === 'expert' && page === 'evaluate' ? <EvaluatePage /> : null}
      </main>
    </div>
  );
}

export default App;
