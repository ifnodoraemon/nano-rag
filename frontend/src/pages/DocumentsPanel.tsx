import { useEffect } from 'react';
import { useAppStore } from '../stores/appStore';
import { Card, LoadingButton, Panel, StatusLine } from '../components/common';

interface DocumentsPanelProps {
  audience?: 'simple' | 'expert';
}

function formatTimestamp(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleString('zh-CN', {
    hour12: false,
  });
}

export function DocumentsPanel({ audience = 'expert' }: DocumentsPanelProps) {
  const {
    workspace,
    documents,
    documentsLoading,
    documentsError,
    loadDocuments,
  } = useAppStore();

  useEffect(() => {
    void loadDocuments();
  }, [loadDocuments, workspace.apiKey, workspace.kbId, workspace.tenantId]);

  return (
    <Panel
      title={audience === 'simple' ? '当前资料范围' : '当前工作区资料'}
      subtitle={
        audience === 'simple'
          ? '这里显示当前测试空间里已经生效的资料，不再只看“刚上传了什么”'
          : '用来确认当前 kb / tenant 下实际可被检索到的资料范围'
      }
      actions={
        <LoadingButton
          loading={documentsLoading}
          type="button"
          variant="secondary"
          onClick={() => void loadDocuments()}
        >
          刷新资料库
        </LoadingButton>
      }
    >
      <div className="stack">
        <div className="status-tip">
          当前作用域是 `{workspace.kbId}` / `{workspace.tenantId || 'default-tenant'}`。同名上传会覆盖旧版本，不会继续堆积重复 chunk。
        </div>
        <StatusLine
          message={
            documentsLoading
              ? '正在加载资料清单...'
              : documentsError
                ? documentsError
                : documents.length
                  ? `当前共有 ${documents.length} 份资料处于可检索状态。`
                  : '当前作用域下还没有资料。'
          }
          isError={!!documentsError}
        />
        {documents.length ? (
          <div className="cards compact-cards">
            {documents.map((document) => (
              <Card key={document.doc_id} title={document.title}>
                source: {document.source_path}
                {'\n'}chunks: {document.chunk_count}
                {'\n'}updated: {formatTimestamp(document.updated_at)}
                {document.doc_type ? `\ntype: ${document.doc_type}` : ''}
              </Card>
            ))}
          </div>
        ) : (
          <div className="empty-state">
            {audience === 'simple'
              ? '先上传资料，资料库会显示这次测试真正生效的文档范围。'
              : '先导入资料，再用这里核对当前工作区里实际存在的文档。'}
          </div>
        )}
      </div>
    </Panel>
  );
}
