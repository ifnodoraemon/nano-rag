import { useEffect, useMemo, useState } from 'react';
import { useAppStore } from '../stores/appStore';
import { Panel, StatusLine, JsonOutput, LoadingButton, Card } from '../components/common';

function formatTimestamp(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleString('zh-CN', {
    hour12: false,
  });
}

export function EvalPanel() {
  const {
    workspace,
    evalResult,
    evalLoading,
    evalError,
    evalDatasets,
    evalDatasetsLoading,
    evalDatasetsError,
    loadEvalDatasets,
    evalReports,
    evalReportsLoading,
    evalReportsError,
    loadEvalReports,
    currentEvalReport,
    currentEvalReportPath,
    evalReportLoading,
    evalReportError,
    loadEvalReport,
    benchmarkResult,
    benchmarkLoading,
    benchmarkError,
    benchmarkReports,
    benchmarkReportsLoading,
    benchmarkReportsError,
    loadBenchmarkReports,
    currentBenchmarkReport,
    currentBenchmarkReportPath,
    benchmarkReportLoading,
    benchmarkReportError,
    loadBenchmarkReport,
    runBenchmark,
    diagnosis,
    diagnosisLoading,
    diagnosisError,
    diagnoseEvalResult,
    runEval,
  } = useAppStore();
  const [datasetPath, setDatasetPath] = useState('data/eval/employee_handbook_eval.jsonl');
  const [outputPath, setOutputPath] = useState('');

  useEffect(() => {
    loadEvalDatasets();
    loadEvalReports();
    loadBenchmarkReports();
  }, [loadBenchmarkReports, loadEvalDatasets, loadEvalReports]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (datasetPath.trim()) {
      runEval(datasetPath.trim(), outputPath.trim() || undefined);
    }
  };

  const activeReport = currentEvalReport || evalResult?.report || null;
  const activeBenchmarkReport = currentBenchmarkReport || benchmarkResult?.report || null;
  const aggregate = activeReport?.aggregate;
  const benchmarkAggregate =
    activeBenchmarkReport && typeof activeBenchmarkReport === 'object'
      ? (activeBenchmarkReport.aggregate as Record<string, number> | undefined)
      : undefined;
  const failedCases = useMemo(
    () =>
      (activeReport?.results || [])
        .map((item, reportIndex) => ({ item, reportIndex }))
        .filter(
          ({ item }) => item.answer_exact_match < 1 || item.reference_context_recall < 1,
        ),
    [activeReport],
  );

  return (
    <Panel
      title="离线评测"
      subtitle="选择数据集，运行评测，并查看聚合指标与坏例"
    >
      <form onSubmit={handleSubmit}>
        <div className="status-tip">
          评测会直接调用当前后端的 RAG 链路。如果数据集里没有预填 `answer` 或
          `retrieved_contexts`，系统会自动跑一遍当前模型与检索流程再计算指标。
          当前工作区是 `{workspace.kbId}` / `{workspace.tenantId || 'default-tenant'}`；
          如果数据集样本本身带 `kb_id/tenant_id/session_id`，会覆盖这里的默认值。
        </div>

        <div className="section-label">可用数据集</div>
        <StatusLine
          message={evalDatasetsLoading ? '正在加载数据集...' : evalDatasetsError || undefined}
          isError={!!evalDatasetsError}
        />
        <div className="cards">
          {evalDatasets.length ? (
            evalDatasets.map((dataset) => (
              <button
                key={dataset.path}
                type="button"
                className={`trace-item secondary${datasetPath === dataset.path ? ' selected-card' : ''}`}
                onClick={() => setDatasetPath(dataset.path)}
              >
                <strong>{dataset.name}</strong>
                <div className="mono">{dataset.path}</div>
                <div className="muted">
                  records={dataset.records} | updated={formatTimestamp(dataset.updated_at)}
                </div>
                {dataset.sample_queries.length > 0 && (
                  <div className="muted">示例: {dataset.sample_queries.join(' / ')}</div>
                )}
              </button>
            ))
          ) : (
            <div className="empty-state">当前没有可用评测集。默认会从 `data/eval/*.jsonl` 扫描。</div>
          )}
        </div>

        <label>
          数据集路径（JSONL）
          <input
            value={datasetPath}
            onChange={(e) => setDatasetPath(e.target.value)}
            placeholder="./data/eval/employee_handbook_eval.jsonl"
          />
        </label>
        <label>
          输出路径（可选）
          <input
            value={outputPath}
            onChange={(e) => setOutputPath(e.target.value)}
            placeholder="留空则自动保存到 data/reports/eval/"
          />
        </label>
        <div className="actions">
          <LoadingButton loading={evalLoading} type="submit">
            开始评测
          </LoadingButton>
          <LoadingButton
            loading={benchmarkLoading}
            type="button"
            variant="secondary"
            onClick={() => runBenchmark(datasetPath.trim(), outputPath.trim() || undefined)}
          >
            运行 Benchmark
          </LoadingButton>
        </div>
        <StatusLine
          message={
            evalLoading
              ? '正在执行评测...'
              : evalError
                ? evalError
                : evalResult
                  ? `评测完成。状态: ${evalResult.status} | 输出: ${evalResult.output_path || '未保存'}`
                  : undefined
          }
          isError={!!evalError}
        />
        <StatusLine
          message={
            benchmarkLoading
              ? '正在执行 benchmark...'
              : benchmarkError
                ? benchmarkError
                : benchmarkResult
                  ? `Benchmark 完成。输出: ${benchmarkResult.output_path || '未保存'}`
                  : undefined
          }
          isError={!!benchmarkError}
        />
      </form>

      {aggregate ? (
        <div className="stack" style={{ marginTop: 18 }}>
          <div className="section-label">聚合指标</div>
          <div className="metric-grid">
            <div className="metric-card">
              <span>Answer Exact Match</span>
              <strong>{aggregate.answer_exact_match}</strong>
            </div>
            <div className="metric-card">
              <span>Reference Context Recall</span>
              <strong>{aggregate.reference_context_recall}</strong>
            </div>
            <div className="metric-card">
              <span>Retrieved Context Count Avg</span>
              <strong>{aggregate.retrieved_context_count_avg}</strong>
            </div>
            <div className="metric-card">
              <span>评测记录数</span>
              <strong>{activeReport?.records || 0}</strong>
            </div>
          </div>

          <div>
            <div className="section-label">坏例与待检查样本</div>
            <div className="cards">
              {failedCases.length ? (
                failedCases.map(({ item, reportIndex }, index) => (
                  <Card
                    key={`${item.query}-${index}`}
                    title={item.query || `样本 ${index + 1}`}
                  >
                    exact_match={item.answer_exact_match} | context_recall={item.reference_context_recall}
                    {'\n'}
                    trace_id: {item.trace_id || 'n/a'}
                    {'\n'}
                    answer: {item.answer || 'n/a'}
                    {'\n'}
                    reference: {item.reference_answer || 'n/a'}
                    {'\n'}
                    {currentEvalReportPath && (
                      <button
                        type="button"
                        className="secondary"
                        onClick={() => diagnoseEvalResult(currentEvalReportPath, reportIndex, false)}
                      >
                        诊断该坏例
                      </button>
                    )}
                  </Card>
                ))
              ) : (
                <div className="empty-state">当前这次评测没有发现坏例，所有样本都命中了当前指标阈值。</div>
              )}
            </div>
          </div>

          <StatusLine
            message={
              diagnosisLoading
                ? '正在分析坏例...'
                : diagnosisError
                  ? diagnosisError
                  : diagnosis
                    ? diagnosis.summary
                    : undefined
            }
            isError={!!diagnosisError}
          />
          <JsonOutput data={diagnosis} placeholder="选择坏例后可查看诊断结果" />

          <details className="details-panel">
            <summary>查看完整评测报告</summary>
            <JsonOutput data={activeReport} placeholder="还没有执行评测" />
          </details>
        </div>
      ) : (
        <div className="empty-state" style={{ marginTop: 18 }}>
          选择一个数据集后运行评测。这里会展示聚合指标、坏例列表和完整报告。
        </div>
      )}

      <div className="stack" style={{ marginTop: 18 }}>
        <div className="section-label">历史报告</div>
        <StatusLine
          message={
            evalReportsLoading
              ? '正在加载历史报告...'
              : evalReportLoading
                ? '正在加载报告详情...'
                : evalReportsError || evalReportError || undefined
          }
          isError={!!evalReportsError || !!evalReportError}
        />
        <div className="cards">
          {evalReports.length ? (
            evalReports.slice(0, 8).map((report) => (
              <button
                key={report.path}
                type="button"
                className={`trace-item secondary${currentEvalReportPath === report.path ? ' selected-card' : ''}`}
                onClick={() => loadEvalReport(report.path)}
              >
                <strong>{report.name}</strong>
                <div className="mono">{report.path}</div>
                <div className="muted">
                  records={report.records} | exact_match={report.aggregate.answer_exact_match ?? 'n/a'} |
                  context_recall={report.aggregate.reference_context_recall ?? 'n/a'}
                </div>
                <div className="muted">updated={formatTimestamp(report.updated_at)}</div>
              </button>
            ))
          ) : (
            <div className="empty-state">还没有历史报告。运行一次评测后会自动保存到 `data/reports/eval/`。</div>
          )}
        </div>
      </div>

      <div className="stack" style={{ marginTop: 18 }}>
        <div className="section-label">Benchmark</div>
        <StatusLine
          message={
            benchmarkReportsLoading
              ? '正在加载 benchmark 报告...'
              : benchmarkReportLoading
                ? '正在加载 benchmark 详情...'
                : benchmarkReportsError || benchmarkReportError || undefined
          }
          isError={!!benchmarkReportsError || !!benchmarkReportError}
        />
        {benchmarkAggregate ? (
          <div className="metric-grid">
            <div className="metric-card">
              <span>Bad Case Count</span>
              <strong>{benchmarkAggregate.bad_case_count ?? 'n/a'}</strong>
            </div>
            <div className="metric-card">
              <span>Latency Avg</span>
              <strong>{benchmarkAggregate.latency_seconds_avg ?? 'n/a'}</strong>
            </div>
            <div className="metric-card">
              <span>Latency P95</span>
              <strong>{benchmarkAggregate.latency_seconds_p95 ?? 'n/a'}</strong>
            </div>
          </div>
        ) : (
          <div className="empty-state">运行一次 benchmark 后，这里会展示延迟与坏例聚合。</div>
        )}
        <div className="cards">
          {benchmarkReports.length ? (
            benchmarkReports.slice(0, 8).map((report) => (
              <button
                key={report.path}
                type="button"
                className={`trace-item secondary${currentBenchmarkReportPath === report.path ? ' selected-card' : ''}`}
                onClick={() => loadBenchmarkReport(report.path)}
              >
                <strong>{report.name}</strong>
                <div className="mono">{report.path}</div>
                <div className="muted">
                  records={report.records} | bad_cases={report.aggregate.bad_case_count ?? 'n/a'} |
                  latency_p95={report.aggregate.latency_seconds_p95 ?? 'n/a'}
                </div>
                <div className="muted">updated={formatTimestamp(report.updated_at)}</div>
              </button>
            ))
          ) : (
            <div className="empty-state">还没有 benchmark 报告。点击“运行 Benchmark”后会自动保存。</div>
          )}
        </div>
        <details className="details-panel">
          <summary>查看完整 benchmark 报告</summary>
          <JsonOutput data={activeBenchmarkReport} placeholder="还没有 benchmark 报告" />
        </details>
      </div>
    </Panel>
  );
}
