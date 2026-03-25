import { useState } from 'react';
import { useAppStore } from '../stores/appStore';
import { Panel, StatusLine, JsonOutput, LoadingButton } from '../components/common';

export function IngestPanel() {
  const { workspace, ingestResult, ingestLoading, ingestError, runIngest } = useAppStore();
  const [path, setPath] = useState('./data/raw');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    runIngest(path);
  };

  return (
    <Panel
      title="步骤 1 · 导入测试知识"
      subtitle="先把测试资料导入知识库，再开始问答验证"
    >
      <form onSubmit={handleSubmit}>
        <div className="status-tip">
          当前后端 ingest 接口仍使用服务器目录路径，因此这里填写的是服务端可访问的目录，不是浏览器本地文件路径。
          本次导入会写入当前工作区：`{workspace.kbId}` / `{workspace.tenantId || 'default-tenant'}`。
        </div>
        <label>
          测试资料目录
          <input
            value={path}
            onChange={(e) => setPath(e.target.value)}
            placeholder="./data/raw"
          />
        </label>
        <div className="actions">
          <LoadingButton loading={ingestLoading} type="submit">
            开始导入
          </LoadingButton>
          <button
            type="button"
            className="secondary"
            onClick={() => setPath('./data/raw')}
          >
            使用示例目录
          </button>
        </div>
        <StatusLine
          message={ingestLoading ? '正在导入测试资料...' : ingestError || undefined}
          isError={!!ingestError}
        />
        {ingestResult ? (
          <div className="metric-grid">
            <div className="metric-card">
              <span>导入文档</span>
              <strong>{ingestResult.documents}</strong>
            </div>
            <div className="metric-card">
              <span>生成分片</span>
              <strong>{ingestResult.chunks}</strong>
            </div>
            <div className="metric-card">
              <span>知识库</span>
              <strong>{ingestResult.kb_id || workspace.kbId}</strong>
            </div>
          </div>
        ) : (
          <div className="empty-state">还没有执行导入。默认可以先用 `./data/raw` 做联调。</div>
        )}
        <details className="details-panel">
          <summary>查看导入返回</summary>
          <JsonOutput data={ingestResult} placeholder="Ingest not run yet" />
        </details>
      </form>
    </Panel>
  );
}
