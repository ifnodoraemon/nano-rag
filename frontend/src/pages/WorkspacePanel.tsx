import { useState } from 'react';
import { useAppStore } from '../stores/appStore';
import { Panel, StatusLine } from '../components/common';

export function WorkspacePanel() {
  const { workspace, updateWorkspace } = useAppStore();
  const [draft, setDraft] = useState(workspace);
  const [message, setMessage] = useState<string | null>(null);

  const updateField = (field: keyof typeof draft, value: string) => {
    setDraft((current) => ({ ...current, [field]: value }));
    setMessage(null);
  };

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    const next = {
      kbId: draft.kbId.trim() || 'default',
      tenantId: draft.tenantId.trim(),
      sessionId: draft.sessionId.trim(),
      apiKey: draft.apiKey.trim(),
    };
    updateWorkspace(next);
    setDraft(next);
    setMessage('工作区配置已保存。主流程会直接使用 `/v1/rag/*` 业务接口。');
  };

  return (
    <Panel
      title="工作区配置"
      subtitle="统一管理知识库、租户、会话和业务 API Key"
    >
      <form onSubmit={handleSubmit}>
        <div className="status-tip">
          NanoRAG 的主流程前端现在默认走正式业务 API，而不是内部 `/chat` 或 `/ingest`。
          如果后端启用了 `RAG_API_KEYS`，请把对应 key 填在这里。出于安全考虑，页面不会把这个 key 持久化到本地存储。
        </div>
        <div className="workspace-grid">
          <label>
            kb_id
            <input
              value={draft.kbId}
              onChange={(event) => updateField('kbId', event.target.value)}
              placeholder="default"
            />
          </label>
          <label>
            tenant_id
            <input
              value={draft.tenantId}
              onChange={(event) => updateField('tenantId', event.target.value)}
              placeholder="demo-tenant"
            />
          </label>
          <label>
            session_id
            <input
              value={draft.sessionId}
              onChange={(event) => updateField('sessionId', event.target.value)}
              placeholder="session-web"
            />
          </label>
          <label>
            业务 API Key
            <input
              type="password"
              value={draft.apiKey}
              onChange={(event) => updateField('apiKey', event.target.value)}
              placeholder="留空表示当前环境未启用业务鉴权"
            />
          </label>
        </div>
        <div className="actions">
          <button type="submit" className="primary">
            保存工作区
          </button>
        </div>
        <StatusLine message={message || undefined} isError={false} />
      </form>
    </Panel>
  );
}
