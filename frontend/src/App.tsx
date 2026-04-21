import { useEffect, useState } from 'react';
import { Chip } from './components/common';
import {
  getPageFromHash,
  getStoredAudienceMode,
  navigateToPage,
  type AppPage,
  type AudienceMode,
} from './navigation';
import { SimpleHomePage } from './pages/SimpleHomePage';
import { ValidatePage } from './pages/ValidatePage';
import { InvestigatePage } from './pages/InvestigatePage';
import { EvaluatePage } from './pages/EvaluatePage';
import { useAppStore } from './stores/appStore';
import './index.css';

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
  const [mode, setMode] = useState<AudienceMode>(() => getStoredAudienceMode());
  const { workspace, health, healthLoading, loadHealth, ingestResult, chatResult } = useAppStore();

  useEffect(() => {
    void loadHealth();
  }, [loadHealth]);

  useEffect(() => {
    window.localStorage.setItem('nano-rag-mode', mode);
  }, [mode]);

  useEffect(() => {
    const onHashChange = () => {
      setPage(getPageFromHash(window.location.hash));
    };
    const onModeChange = (event: Event) => {
      const customEvent = event as CustomEvent<AudienceMode>;
      setMode(customEvent.detail === 'expert' ? 'expert' : 'simple');
    };
    window.addEventListener('hashchange', onHashChange);
    window.addEventListener('nano-rag:set-mode', onModeChange as EventListener);
    return () => {
      window.removeEventListener('hashchange', onHashChange);
      window.removeEventListener('nano-rag:set-mode', onModeChange as EventListener);
    };
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
  const evaluateAvailable = !!(health?.features?.eval || health?.features?.benchmark);
  const expertPages: AppPage[] = evaluateAvailable
    ? ['validate', 'investigate', 'evaluate']
    : ['validate', 'investigate'];
  const nextStep = !health
    ? '等待系统健康检查'
    : health.status !== 'ok'
      ? '先修复运行状态，再开始导入'
      : !ingestResult
        ? '先上传一批资料'
        : !chatResult
          ? '下一步直接提一个真实问题'
          : '继续复核引用，必要时进入排查页';

  useEffect(() => {
    if (page === 'evaluate' && !evaluateAvailable) {
      navigateToPage('validate');
    }
  }, [evaluateAvailable, page]);

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
            {expertPages.map((targetPage) => (
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
              <span>推荐下一步</span>
              <strong>{nextStep}</strong>
              <p>让第一次体验更像产品流程，而不是先学一堆内部名词。</p>
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
            <div className="summary-card">
              <span>工作台范围</span>
              <strong>
                {health?.features?.eval || health?.features?.benchmark ? 'Core + Workbench' : 'Nano Core'}
              </strong>
              <p>
                {health?.features?.eval || health?.features?.benchmark
                  ? '当前实例开放了评测/诊断扩展能力。'
                  : '当前实例只暴露核心验证链路，更接近轻量产品形态。'}
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
