import { useEffect, useMemo, useState } from 'react';
import { useAppStore, type EvalClaimFilter } from '../stores/appStore';
import { Panel, StatusLine, JsonOutput, LoadingButton, Card } from '../components/common';

function openAdvancedPanel(targetId: string): void {
  const advanced = document.getElementById('advanced-workbench');
  if (advanced instanceof HTMLDetailsElement) {
    advanced.open = true;
  }
  window.requestAnimationFrame(() => {
    document.getElementById(targetId)?.scrollIntoView({
      behavior: 'smooth',
      block: 'start',
    });
  });
}

function formatTimestamp(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleString('zh-CN', {
    hour12: false,
  });
}

function formatMetric(value: number | undefined): string {
  return value === undefined ? 'n/a' : String(value);
}

function formatRate(value: number | undefined): string {
  return value === undefined ? 'n/a' : `${(value * 100).toFixed(1)}%`;
}

function matchesClaimFilter(
  item: {
    conflicting_context_count?: number;
    conflict_claim_count?: number;
    insufficiency_claim_count?: number;
  },
  claimFilter: EvalClaimFilter,
): boolean {
  if (claimFilter === 'missing_conflict') {
    return (item.conflicting_context_count ?? 0) > 0 && (item.conflict_claim_count ?? 0) === 0;
  }
  if (claimFilter === 'insufficiency') {
    return (item.insufficiency_claim_count ?? 0) > 0;
  }
  return true;
}

function claimFilterLabel(claimFilter: EvalClaimFilter): string {
  if (claimFilter === 'missing_conflict') {
    return '只看缺少 conflict claim';
  }
  if (claimFilter === 'insufficiency') {
    return '只看 insufficiency claims';
  }
  return '全部 claims';
}

function getBenchmarkCaseKey(item: Record<string, unknown>, index: number): string {
  return String(item.trace_id || item.query || `benchmark-${index}`);
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
    selectedEvalResultIndex,
    evalReportLoading,
    evalReportError,
    loadEvalReport,
    setSelectedEvalResultIndex,
    benchmarkResult,
    benchmarkLoading,
    benchmarkError,
    benchmarkReports,
    benchmarkReportsLoading,
    benchmarkReportsError,
    loadBenchmarkReports,
    currentBenchmarkReport,
    currentBenchmarkReportPath,
    selectedBenchmarkCaseKey,
    benchmarkReportLoading,
    benchmarkReportError,
    loadBenchmarkReport,
    setSelectedBenchmarkCaseKey,
    runBenchmark,
    diagnosis,
    diagnosisLoading,
    diagnosisError,
    diagnoseEvalResult,
    runEval,
    evalConflictOnly,
    benchmarkConflictOnly,
    evalClaimFilter,
    benchmarkClaimFilter,
    setEvalConflictOnly,
    setBenchmarkConflictOnly,
    setEvalClaimFilter,
    setBenchmarkClaimFilter,
    setSelectedTraceId,
    prepareChatReplay,
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
  const benchmarkResults =
    activeBenchmarkReport &&
    typeof activeBenchmarkReport === 'object' &&
    Array.isArray((activeBenchmarkReport as Record<string, unknown>).results)
      ? ((activeBenchmarkReport as Record<string, unknown>).results as Record<string, unknown>[])
      : [];
  const failedCases = useMemo(
    () =>
      (activeReport?.results || [])
        .map((item, reportIndex) => ({ item, reportIndex }))
        .filter(
          ({ item }) => item.answer_exact_match < 1 || item.reference_context_recall < 1,
        ),
    [activeReport],
  );
  const benchmarkBadCases = useMemo(
    () =>
      benchmarkResults.filter(
        (item) =>
          Number(item.answer_exact_match ?? 1) < 1 || Number(item.reference_context_recall ?? 1) < 1,
      ),
    [benchmarkResults],
  );
  const benchmarkConflictCases = useMemo(
    () =>
      benchmarkBadCases.filter((item) => Number(item.conflicting_context_count ?? 0) > 0),
    [benchmarkBadCases],
  );
  const visibleFailedCases = useMemo(
    () =>
      failedCases.filter(({ item }) => {
        if (evalConflictOnly && (item.conflicting_context_count ?? 0) === 0) {
          return false;
        }
        return matchesClaimFilter(item, evalClaimFilter);
      }),
    [evalClaimFilter, evalConflictOnly, failedCases],
  );
  const visibleEvalReports = useMemo(
    () =>
      evalConflictOnly
        ? evalReports.filter((report) => Number(report.aggregate.conflicting_hit_rate ?? 0) > 0)
        : evalReports,
    [evalConflictOnly, evalReports],
  );
  const visibleBenchmarkBadCases = useMemo(
    () =>
      benchmarkBadCases.filter((item) => {
        if (benchmarkConflictOnly && Number(item.conflicting_context_count ?? 0) === 0) {
          return false;
        }
        return matchesClaimFilter(
          {
            conflicting_context_count: Number(item.conflicting_context_count ?? 0),
            conflict_claim_count: Number(item.conflict_claim_count ?? 0),
            insufficiency_claim_count: Number(item.insufficiency_claim_count ?? 0),
          },
          benchmarkClaimFilter,
        );
      }),
    [benchmarkBadCases, benchmarkClaimFilter, benchmarkConflictOnly],
  );
  const visibleBenchmarkReports = useMemo(
    () =>
      benchmarkConflictOnly
        ? benchmarkReports.filter(
            (report) =>
              Number(report.aggregate.conflicting_bad_case_count ?? 0) > 0 ||
              Number(report.aggregate.conflicting_hit_rate ?? 0) > 0,
          )
        : benchmarkReports,
    [benchmarkConflictOnly, benchmarkReports],
  );
  const selectedBenchmarkCase = useMemo(
    () =>
      benchmarkBadCases.find(
        (item, index) => getBenchmarkCaseKey(item, index) === selectedBenchmarkCaseKey,
      ) || null,
    [benchmarkBadCases, selectedBenchmarkCaseKey],
  );
  const selectedBenchmarkDiagnosis =
    selectedBenchmarkCase &&
    typeof selectedBenchmarkCase.diagnosis === 'object' &&
    selectedBenchmarkCase.diagnosis !== null
      ? (selectedBenchmarkCase.diagnosis as Record<string, unknown>)
      : null;

  useEffect(() => {
    if (selectedEvalResultIndex === null) {
      return;
    }
    window.requestAnimationFrame(() => {
      document
        .getElementById(`eval-result-${selectedEvalResultIndex}`)
        ?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
  }, [selectedEvalResultIndex, visibleFailedCases.length]);

  useEffect(() => {
    if (!selectedBenchmarkCaseKey) {
      return;
    }
    window.requestAnimationFrame(() => {
      document
        .getElementById(`benchmark-case-${selectedBenchmarkCaseKey}`)
        ?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
  }, [selectedBenchmarkCaseKey, visibleBenchmarkBadCases.length]);

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
          <button
            type="button"
            className="secondary"
            onClick={() => setEvalConflictOnly(!evalConflictOnly)}
          >
            {evalConflictOnly ? '评测显示全部' : '评测只看冲突'}
          </button>
          <button
            type="button"
            className="secondary"
            onClick={() => setBenchmarkConflictOnly(!benchmarkConflictOnly)}
          >
            {benchmarkConflictOnly ? 'Benchmark 显示全部' : 'Benchmark 只看冲突'}
          </button>
        </div>
        <div className="actions">
          <button
            type="button"
            className={`secondary${evalClaimFilter === 'all' ? ' selected-card' : ''}`}
            onClick={() => setEvalClaimFilter('all')}
          >
            Eval 全部 claims
          </button>
          <button
            type="button"
            className={`secondary${evalClaimFilter === 'missing_conflict' ? ' selected-card' : ''}`}
            onClick={() => setEvalClaimFilter('missing_conflict')}
          >
            Eval 缺少 conflict claim
          </button>
          <button
            type="button"
            className={`secondary${evalClaimFilter === 'insufficiency' ? ' selected-card' : ''}`}
            onClick={() => setEvalClaimFilter('insufficiency')}
          >
            Eval insufficiency
          </button>
          <button
            type="button"
            className={`secondary${benchmarkClaimFilter === 'all' ? ' selected-card' : ''}`}
            onClick={() => setBenchmarkClaimFilter('all')}
          >
            Benchmark 全部 claims
          </button>
          <button
            type="button"
            className={`secondary${benchmarkClaimFilter === 'missing_conflict' ? ' selected-card' : ''}`}
            onClick={() => setBenchmarkClaimFilter('missing_conflict')}
          >
            Benchmark 缺少 conflict claim
          </button>
          <button
            type="button"
            className={`secondary${benchmarkClaimFilter === 'insufficiency' ? ' selected-card' : ''}`}
            onClick={() => setBenchmarkClaimFilter('insufficiency')}
          >
            Benchmark insufficiency
          </button>
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
              <span>Conflicting Context Avg</span>
              <strong>{formatMetric(aggregate.conflicting_context_count_avg)}</strong>
            </div>
            <div className="metric-card">
              <span>Conflict Hit Rate</span>
              <strong>{formatRate(aggregate.conflicting_hit_rate)}</strong>
            </div>
            <div className="metric-card">
              <span>Conflict Claim Hit Rate</span>
              <strong>{formatRate(aggregate.conflict_claim_hit_rate)}</strong>
            </div>
            <div className="metric-card">
              <span>Insufficiency Claim Hit Rate</span>
              <strong>{formatRate(aggregate.insufficiency_claim_hit_rate)}</strong>
            </div>
            <div className="metric-card">
              <span>评测记录数</span>
              <strong>{activeReport?.records || 0}</strong>
            </div>
          </div>

          <div>
            <div className="section-label">坏例与待检查样本</div>
            {(evalConflictOnly || evalClaimFilter !== 'all') && (
              <div className="status-line">
                当前 Eval 筛选:
                {evalConflictOnly ? ' 只看冲突' : ' 全部坏例'} | {claimFilterLabel(evalClaimFilter)}
              </div>
            )}
            <div className="cards">
              {visibleFailedCases.length ? (
                visibleFailedCases.map(({ item, reportIndex }, index) => (
                  <Card
                    key={`${item.query}-${index}`}
                    title={item.query || `样本 ${index + 1}`}
                    id={`eval-result-${reportIndex}`}
                    className={selectedEvalResultIndex === reportIndex ? 'selected-card' : undefined}
                  >
                    exact_match={item.answer_exact_match} | context_recall={item.reference_context_recall}
                    {'\n'}
                    conflicts={item.conflicting_context_count ?? 0}
                    {'\n'}
                    conflict_claims={item.conflict_claim_count ?? 0} | insufficiency_claims=
                    {item.insufficiency_claim_count ?? 0}
                    {Number(item.conflicting_context_count ?? 0) > 0 &&
                    Number(item.conflict_claim_count ?? 0) === 0
                      ? '\nmissing conflict claim'
                      : ''}
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
                        onClick={() => {
                          setSelectedEvalResultIndex(reportIndex);
                          void diagnoseEvalResult(currentEvalReportPath, reportIndex, false);
                        }}
                      >
                        诊断该坏例
                      </button>
                    )}
                    <button
                      type="button"
                      className="secondary"
                      onClick={() => setSelectedEvalResultIndex(reportIndex)}
                    >
                      聚焦该样本
                    </button>
                    {item.trace_id && (
                      <button
                        type="button"
                        className="secondary"
                        onClick={() => {
                          setSelectedTraceId(String(item.trace_id));
                          openAdvancedPanel('traces-panel');
                        }}
                      >
                        打开 Trace
                      </button>
                    )}
                    {item.query && (
                      <button
                        type="button"
                        className="secondary"
                        onClick={() =>
                          prepareChatReplay({
                            query: item.query || '',
                            traceId: item.trace_id || undefined,
                            sourceLabel: `eval sample ${reportIndex}`,
                          })
                        }
                      >
                        回放到 Chat
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
          {visibleEvalReports.length ? (
            visibleEvalReports.slice(0, 8).map((report) => (
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
                <div className="muted">
                  conflict_hit_rate={formatRate(report.aggregate.conflicting_hit_rate)} |
                  conflict_avg={formatMetric(report.aggregate.conflicting_context_count_avg)}
                </div>
                <div className="muted">
                  conflict_claim_hit_rate={formatRate(report.aggregate.conflict_claim_hit_rate)} |
                  insufficiency_claim_hit_rate=
                  {formatRate(report.aggregate.insufficiency_claim_hit_rate)}
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
          <div className="stack">
            <div className="metric-grid">
              <div className="metric-card">
                <span>Bad Case Count</span>
                <strong>{benchmarkAggregate.bad_case_count ?? 'n/a'}</strong>
              </div>
              <div className="metric-card">
                <span>Conflict Bad Cases</span>
                <strong>{benchmarkAggregate.conflicting_bad_case_count ?? 'n/a'}</strong>
              </div>
              <div className="metric-card">
                <span>Conflict Hit Rate</span>
                <strong>{formatRate(benchmarkAggregate.conflicting_hit_rate)}</strong>
              </div>
              <div className="metric-card">
                <span>Conflicting Context Avg</span>
                <strong>{formatMetric(benchmarkAggregate.conflicting_context_count_avg)}</strong>
              </div>
              <div className="metric-card">
                <span>Conflict Claim Hit Rate</span>
                <strong>{formatRate(benchmarkAggregate.conflict_claim_hit_rate)}</strong>
              </div>
              <div className="metric-card">
                <span>Insufficiency Claim Hit Rate</span>
                <strong>{formatRate(benchmarkAggregate.insufficiency_claim_hit_rate)}</strong>
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

            {selectedBenchmarkCaseKey && (
              <div>
                <div className="section-label">当前聚焦 Benchmark Diagnosis</div>
                {selectedBenchmarkDiagnosis ? (
                  <Card
                    title={String(
                      selectedBenchmarkCase?.query ||
                        selectedBenchmarkCase?.trace_id ||
                        'Selected benchmark case',
                    )}
                  >
                    summary: {String(selectedBenchmarkDiagnosis.summary || 'n/a')}
                    {'\n'}
                    trace_id: {String(selectedBenchmarkCase?.trace_id || 'n/a')}
                    {'\n'}
                    severity: {String(selectedBenchmarkDiagnosis.severity || 'n/a')}
                    {selectedBenchmarkCase?.trace_id ? (
                      <>
                        {'\n'}
                        <button
                          type="button"
                          className="secondary"
                          onClick={() => {
                            setSelectedTraceId(String(selectedBenchmarkCase.trace_id));
                            openAdvancedPanel('traces-panel');
                          }}
                        >
                          打开 Trace
                        </button>
                      </>
                    ) : null}
                    {selectedBenchmarkCase?.query ? (
                      <>
                        {'\n'}
                        <button
                          type="button"
                          className="secondary"
                          onClick={() =>
                            prepareChatReplay({
                              query: String(selectedBenchmarkCase.query || ''),
                              traceId: String(selectedBenchmarkCase.trace_id || '') || undefined,
                              sourceLabel: 'selected benchmark case',
                            })
                          }
                        >
                          回放到 Chat
                        </button>
                      </>
                    ) : null}
                  </Card>
                ) : (
                  <div className="empty-state">
                    当前聚焦的 benchmark bad case 没有内嵌 diagnosis 摘要。
                  </div>
                )}
              </div>
            )}

            <div>
              <div className="section-label">Benchmark 坏例</div>
              {(benchmarkConflictOnly || benchmarkClaimFilter !== 'all') && (
                <div className="status-line">
                  当前 Benchmark 筛选:
                  {benchmarkConflictOnly ? ' 只看冲突' : ' 全部坏例'} | {claimFilterLabel(benchmarkClaimFilter)}
                </div>
              )}
              <div className="cards">
                {visibleBenchmarkBadCases.length ? (
                  visibleBenchmarkBadCases.slice(0, 6).map((item, index) => (
                    <Card
                      key={`${String(item.trace_id || item.query || 'benchmark')}-${index}`}
                      title={String(item.query || `坏例 ${index + 1}`)}
                      id={`benchmark-case-${getBenchmarkCaseKey(item, index)}`}
                      className={
                        selectedBenchmarkCaseKey === getBenchmarkCaseKey(item, index)
                          ? 'selected-card'
                          : undefined
                      }
                    >
                      exact_match={String(item.answer_exact_match ?? 'n/a')} | context_recall=
                      {String(item.reference_context_recall ?? 'n/a')}
                      {'\n'}
                      conflicts={String(item.conflicting_context_count ?? 0)} | latency=
                      {String(item.latency_seconds ?? 'n/a')}
                      {'\n'}
                      conflict_claims={String(item.conflict_claim_count ?? 0)} |
                      insufficiency_claims={String(item.insufficiency_claim_count ?? 0)}
                      {'\n'}
                      trace_id: {String(item.trace_id || 'n/a')}
                      {'\n'}
                      answer: {String(item.answer || 'n/a')}
                      {'\n'}
                      <button
                        type="button"
                        className="secondary"
                        onClick={() => setSelectedBenchmarkCaseKey(getBenchmarkCaseKey(item, index))}
                      >
                        聚焦该样本
                      </button>
                      {item.trace_id ? (
                        <button
                          type="button"
                          className="secondary"
                          onClick={() => {
                            setSelectedTraceId(String(item.trace_id));
                            openAdvancedPanel('traces-panel');
                          }}
                        >
                          打开 Trace
                        </button>
                      ) : null}
                      {item.query ? (
                        <button
                          type="button"
                          className="secondary"
                          onClick={() =>
                            prepareChatReplay({
                              query: String(item.query || ''),
                              traceId: item.trace_id ? String(item.trace_id) : undefined,
                              sourceLabel: `benchmark case ${index + 1}`,
                            })
                          }
                        >
                          回放到 Chat
                        </button>
                      ) : null}
                    </Card>
                  ))
                ) : (
                  <div className="empty-state">当前 benchmark 没有坏例。</div>
                )}
              </div>
            </div>

            <div>
              <div className="section-label">冲突命中坏例</div>
              <div className="cards">
                {benchmarkConflictCases.length ? (
                  benchmarkConflictCases.slice(0, 4).map((item, index) => (
                    <Card
                      key={`${String(item.trace_id || item.query || 'conflict')}-conflict-${index}`}
                      title={String(item.query || `冲突坏例 ${index + 1}`)}
                    >
                      conflicts={String(item.conflicting_context_count ?? 0)}
                      {'\n'}
                      trace_id: {String(item.trace_id || 'n/a')}
                      {'\n'}
                      diagnosis: {String(
                        (
                          (item.diagnosis as Record<string, unknown> | undefined)?.summary || 'n/a'
                        ),
                      )}
                    </Card>
                  ))
                ) : (
                  <div className="empty-state">当前 benchmark 没有命中冲突知识的坏例。</div>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="empty-state">运行一次 benchmark 后，这里会展示延迟与坏例聚合。</div>
        )}
        <div className="cards">
          {visibleBenchmarkReports.length ? (
            visibleBenchmarkReports.slice(0, 8).map((report) => (
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
                <div className="muted">
                  conflict_bad_cases={report.aggregate.conflicting_bad_case_count ?? 'n/a'} |
                  conflict_hit_rate={formatRate(report.aggregate.conflicting_hit_rate)}
                </div>
                <div className="muted">
                  conflict_claim_hit_rate={formatRate(report.aggregate.conflict_claim_hit_rate)} |
                  insufficiency_claim_hit_rate=
                  {formatRate(report.aggregate.insufficiency_claim_hit_rate)}
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
