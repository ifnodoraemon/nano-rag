import { useEffect, useMemo, useRef, useState } from 'react';
import { useAppStore } from '../stores/appStore';
import { Panel, StatusLine, JsonOutput, LoadingButton, Card } from '../components/common';

interface IngestPanelProps {
  audience?: 'simple' | 'expert';
}

const ACCEPTED_TYPES = '.pdf,.md,.txt,.html,.png,.jpg,.jpeg,.webp';

export function IngestPanel({ audience = 'expert' }: IngestPanelProps) {
  const {
    workspace,
    updateWorkspace,
    ingestResult,
    ingestLoading,
    ingestError,
    runIngest,
    runIngestUpload,
  } = useAppStore();
  const [path, setPath] = useState('/workspace/data/raw');
  const [files, setFiles] = useState<File[]>([]);
  const [selectionMessage, setSelectionMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (ingestLoading) {
      return;
    }
    if (ingestError) {
      setSelectionMessage('导入没有成功，请检查提示后重试。');
      return;
    }
    if (ingestResult) {
      setSelectionMessage('资料已导入完成，可以继续提问了。');
    }
  }, [ingestLoading, ingestError, ingestResult]);

  const uploadedLabel = useMemo(() => {
    if (!files.length) {
      return '拖入文件，或点击选择文件';
    }
    if (files.length === 1) {
      return files[0].name;
    }
    return `已选择 ${files.length} 个文件`;
  }, [files]);

  const handlePathSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    runIngest(path);
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const nextFiles = Array.from(event.target.files || []);
    setFiles(nextFiles);
    if (!nextFiles.length) {
      setSelectionMessage('还没有选择文件。');
      return;
    }
    if (audience === 'simple') {
      const uploadScopeId = `quick-${Date.now().toString(36)}`;
      updateWorkspace({
        tenantId: uploadScopeId,
        sessionId: `${uploadScopeId}-session`,
      });
      setSelectionMessage(`已选择 ${nextFiles.length} 个文件，正在创建新的测试空间并开始导入...`);
      void runIngestUpload(nextFiles);
      return;
    }
    setSelectionMessage(`已选择 ${nextFiles.length} 个文件，点击“上传并导入”开始处理。`);
  };

  const handleUploadSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!files.length) {
      setSelectionMessage('请先选择至少一个文件。');
      return;
    }
    setSelectionMessage(`已选择 ${files.length} 个文件，正在开始导入...`);
    void runIngestUpload(files);
  };

  return (
    <Panel
      title={audience === 'simple' ? '步骤 1 · 上传资料' : '导入测试知识'}
      subtitle={
        audience === 'simple'
          ? '默认直接上传文件做测试，不需要先理解服务器目录'
          : '默认支持直接上传，也保留服务器目录导入给工程排查使用'
      }
    >
      <div className="stack">
        <form onSubmit={handleUploadSubmit}>
          <div className="status-tip">
            {audience === 'simple'
              ? '把你想测试的资料直接上传进来，然后就可以继续提问。支持 PDF、Markdown、TXT 和 HTML。'
              : '建议优先用文件上传做快速验证。导入结果会写入当前工作区。'}
          </div>

          <label className="upload-dropzone">
            <span className="upload-title">{uploadedLabel}</span>
            <span className="upload-subtitle">
              支持 {ACCEPTED_TYPES.replaceAll(',', ' / ')}，可一次上传多份资料
            </span>
            <input
              ref={fileInputRef}
              type="file"
              accept={ACCEPTED_TYPES}
              multiple
              onChange={handleFileChange}
              className="upload-input"
            />
          </label>

          <StatusLine
            message={
              ingestLoading
                ? '正在导入资料...'
                : ingestError
                  ? '导入没有成功，请检查下面的提示后重试。'
                  : selectionMessage || undefined
            }
            isError={!!ingestError}
          />

          {files.length ? (
            <div className="cards compact-cards">
              {files.map((file) => (
                <Card key={`${file.name}-${file.size}`} title={file.name}>
                  size: {Math.max(1, Math.round(file.size / 1024))} KB
                </Card>
              ))}
            </div>
          ) : null}

          <div className="actions">
            {audience === 'expert' ? (
              <LoadingButton loading={ingestLoading} type="submit">
                上传并导入
              </LoadingButton>
            ) : (
              <button
                type="button"
                className="secondary"
                onClick={() => fileInputRef.current?.click()}
              >
                重新选择文件
              </button>
            )}
            {files.length ? (
              <button
                type="button"
                className="secondary"
                onClick={() => {
                  setFiles([]);
                  setSelectionMessage('已清空文件选择。');
                  if (fileInputRef.current) {
                    fileInputRef.current.value = '';
                  }
                }}
              >
                清空文件
              </button>
            ) : null}
          </div>
        </form>

        <StatusLine
          message={ingestLoading ? '正在导入资料...' : ingestError || undefined}
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
              <span>导入方式</span>
              <strong>{ingestResult.source === 'upload' ? '文件上传' : '目录导入'}</strong>
            </div>
            <div className="metric-card">
              <span>知识库</span>
              <strong>{ingestResult.kb_id || workspace.kbId}</strong>
            </div>
          </div>
        ) : (
          <div className="empty-state">
            {audience === 'simple'
              ? '先上传一份资料，然后再问一个具体问题。'
              : '还没有执行导入。可以先上传文件，也可以切换到下方高级目录模式。'}
          </div>
        )}

        {ingestResult?.uploaded_files?.length ? (
          <div>
            <div className="section-label">本次测试资料</div>
            <div className="cards compact-cards">
              {ingestResult.uploaded_files.map((name) => (
                <Card key={name} title={name}>
                  已进入当前测试工作区
                </Card>
              ))}
            </div>
          </div>
        ) : null}

        <details className="details-panel">
          <summary>{audience === 'simple' ? '高级：使用服务器目录导入' : '高级：目录模式与原始返回'}</summary>
          <div className="stack" style={{ marginTop: 12 }}>
            <form onSubmit={handlePathSubmit}>
              <div className="status-tip">
                这里填写的是后端容器里可访问的目录。当前 Docker 示例目录是 `/workspace/data/raw`。
              </div>
              <label>
                测试资料目录
                <input
                  value={path}
                  onChange={(e) => setPath(e.target.value)}
                  placeholder="/workspace/data/raw"
                />
              </label>
              <div className="actions">
                <LoadingButton loading={ingestLoading} type="submit">
                  按目录导入
                </LoadingButton>
                <button
                  type="button"
                  className="secondary"
                  onClick={() => setPath('/workspace/data/raw')}
                >
                  使用示例目录
                </button>
              </div>
            </form>

            <JsonOutput data={ingestResult} placeholder="Ingest not run yet" />
          </div>
        </details>
      </div>
    </Panel>
  );
}
