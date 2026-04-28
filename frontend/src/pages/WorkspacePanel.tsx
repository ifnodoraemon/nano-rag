import { useEffect, useState } from 'react';
import { useAppStore } from '../stores/appStore';
import { Panel, StatusLine } from '../components/common';

interface WorkspacePanelProps {
  audience?: 'simple' | 'expert';
}

export function WorkspacePanel({ audience = 'expert' }: WorkspacePanelProps) {
  const { workspace, updateWorkspace, loadHealth } = useAppStore();
  const [draft, setDraft] = useState(workspace);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    setDraft(workspace);
  }, [workspace]);

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
    setMessage('工作区已保存。接下来可以直接导入资料并提问。');
    void loadHealth();
  };

  return (
    <Panel
      title={audience === 'simple' ? '测试设置' : '工作区配置'}
      subtitle={
        audience === 'simple'
          ? '这部分不是必填，一般直接用默认值就可以'
          : '先把当前验证会落到哪个工作区定下来'
      }
    >
      <form onSubmit={handleSubmit}>
        <div className="status-tip">
          {audience === 'simple'
            ? '大多数情况下只需要填业务 API Key；只有需要隔离资料时，才切换知识库或租户。'
            : '主流程默认调用 `/v1/rag/*` 业务接口。共享或生产环境需要填写 `RAG_API_KEYS` 中的一枚 key；本地显式禁用鉴权时可留空。'}
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
              placeholder="仅在 RAG_AUTH_DISABLED=true 的本地环境可留空"
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
