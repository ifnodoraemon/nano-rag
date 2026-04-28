import { useEffect, useMemo, useState } from 'react';
import { useAppStore, type EvalClaimFilter } from '../stores/appStore';
import {
  Panel,
  StatusLine,
  JsonOutput,
  LoadingButton,
  Card,
  UnavailableState,
} from '../components/common';
import { navigateToPage } from '../navigation';

function formatTimestamp(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleString('zh-CN', {
    hour12: false,
  });
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
    return '只看 insufficiency';
  }
  return '全部样本';
}

function getBenchmarkCaseKey(item: Record<string, unknown>, index: number): string {
  return String(item.trace_id || item.query || `benchmark-${index}`);
}

export function EvalPanel() {
  const {
    health,
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

  const evalEnabled = !!health?.features?.eval;
  const benchmarkEnabled = !!health?.features?.benchmark;
  const diagnosisEnabled = !!health?.features?.diagnosis;

  useEffect(() => {
    if (evalEnabled) {
      loadEvalDatasets();
      loadEvalReports();
    }
    if (benchmarkEnabled) {
      loadBenchmarkReports();
    }
  }, [benchmarkEnabled, evalEnabled, loadBenchmarkReports, loadEvalDatasets, loadEvalReports]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (evalEnabled && datasetPath.trim()) {
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
  const benchmarkResults = useMemo(
    () =>
      activeBenchmarkReport &&
      typeof activeBenchmarkReport === 'object' &&
      Array.isArray((activeBenchmarkReport as Record<string, unknown>).results)
        ? ((activeBenchmarkReport as Record<string, unknown>).results as Record<string, unknown>[])
        : [],
    [activeBenchmarkReport],
  );

  const failedCases = useMemo(
    () =>
      (activeReport?.results || [])
        .map((item, reportIndex) => ({ item, reportIndex }))
        .filter(
          ({ item }) => item.answer_exact_match < 1 || item.reference_context_recall < 1,
        ),
    [activeReport],
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

  const benchmarkBadCases = useMemo(
    () =>
      benchmarkResults.filter(
        (item) =>
          Number(item.answer_exact_match ?? 1) < 1 ||
          Number(item.reference_context_recall ?? 1) < 1,
      ),
    [benchmarkResults],
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

  const selectedBenchmarkCase = useMemo(
    () =>
      benchmarkBadCases.find(
        (item, index) => getBenchmarkCaseKey(item, index) === selectedBenchmarkCaseKey,
      ) || null,
    [benchmarkBadCases, selectedBenchmarkCaseKey],
  );

  const diagnosisSummary =
    selectedEvalResultIndex !== null && diagnosis?.target_type === 'eval'
      ? diagnosis.summary
      : null;

  if (!evalEnabled && !benchmarkEnabled) {
    return (
      <Panel title="离线评测" subtitle="这个实例当前只保留 nano core，不开放离线评测工作台">
        <UnavailableState
          title="Eval / Benchmark 未启用"
          description="当前部署更偏轻量产品形态，只开放导入、问答、trace 查看等核心路径。离线评测和 benchmark 被显式关闭了。"
          hint="启用方式: RAG_EVAL_ENABLED=true；若需要 benchmark 诊断，还需 RAG_DIAGNOSIS_ENABLED=true"
        />
      </Panel>
    );
  }

  return (
    <Panel
      title="离线评测"
      subtitle="只在需要批量验证和坏例分析时打开，默认不压在主流程前面"
    >
      <form onSubmit={handleSubmit}>
        <div className="status-tip">
          当前默认工作区是 `{workspace.kbId}` / `{workspace.tenantId || 'default-tenant'}`。
          如果样本里自带 `kb_id / tenant_id / session_id`，会覆盖这里的默认值。
        </div>
        {!diagnosisEnabled ? (
          <div className="status-line">
            当前实例未启用 diagnosis。你仍然可以跑 Eval，但坏例不会生成自动诊断建议。
          </div>
        ) : null}

        {evalEnabled ? (
          <>
            <div className="section-label">选择数据集</div>
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
                  </button>
                ))
              ) : (
                <div className="empty-state">当前没有可用评测集，会默认扫描 `data/eval/*.jsonl`。</div>
              )}
            </div>
            <label>
              数据集路径
              <input
                value={datasetPath}
                onChange={(e) => setDatasetPath(e.target.value)}
                placeholder="data/eval/employee_handbook_eval.jsonl"
              />
            </label>
          </>
        ) : null}

        <label>
          输出路径（可选）
          <input
            value={outputPath}
            onChange={(e) => setOutputPath(e.target.value)}
            placeholder="留空则自动保存"
          />
        </label>

        <div className="actions">
          {evalEnabled ? (
            <LoadingButton loading={evalLoading} type="submit">
              运行 Eval
            </LoadingButton>
          ) : null}
          {benchmarkEnabled ? (
            <LoadingButton
              loading={benchmarkLoading}
              type="button"
              variant="secondary"
              onClick={() => runBenchmark(datasetPath.trim(), outputPath.trim() || undefined)}
            >
              运行 Benchmark
            </LoadingButton>
          ) : null}
        </div>

        {evalEnabled ? (
          <StatusLine
            message={
              evalLoading
                ? '正在执行评测...'
                : evalError
                  ? evalError
                  : evalResult
                    ? `评测完成。输出: ${evalResult.output_path || '自动保存'}`
                    : undefined
            }
            isError={!!evalError}
          />
        ) : null}
        {benchmarkEnabled ? (
          <StatusLine
            message={
              benchmarkLoading
                ? '正在执行 benchmark...'
                : benchmarkError
                  ? benchmarkError
                  : benchmarkResult
                    ? `Benchmark 完成。输出: ${benchmarkResult.output_path || '自动保存'}`
                    : undefined
            }
            isError={!!benchmarkError}
          />
        ) : null}
      </form>

      {evalEnabled && aggregate ? (
        <div className="stack" style={{ marginTop: 18 }}>
          <div className="section-label">本次评测结果</div>
          <div className="metric-grid">
            <div className="metric-card">
              <span>Answer Exact Match</span>
              <strong>{aggregate.answer_exact_match}</strong>
            </div>
            <div className="metric-card">
              <span>Context Recall</span>
              <strong>{aggregate.reference_context_recall}</strong>
            </div>
            <div className="metric-card">
              <span>Conflict Hit Rate</span>
              <strong>{formatRate(aggregate.conflicting_hit_rate)}</strong>
            </div>
            <div className="metric-card">
              <span>Missing Conflict Claims</span>
              <strong>
                {formatRate(
                  Math.max(
                    (aggregate.conflicting_hit_rate ?? 0) -
                      (aggregate.conflict_claim_hit_rate ?? 0),
                    0,
                  ),
                )}
              </strong>
            </div>
          </div>

          <div>
            <div className="section-label">优先处理的坏例</div>
            {(evalConflictOnly || evalClaimFilter !== 'all') && (
              <div className="status-line">
                当前筛选：{evalConflictOnly ? '只看冲突' : '全部坏例'} | {claimFilterLabel(evalClaimFilter)}
              </div>
            )}
            <div className="cards">
              {visibleFailedCases.length ? (
                visibleFailedCases.slice(0, 6).map(({ item, reportIndex }, index) => (
                  <Card
                    key={`${item.query}-${index}`}
                    title={item.query || `样本 ${index + 1}`}
                    id={`eval-result-${reportIndex}`}
                    className={selectedEvalResultIndex === reportIndex ? 'selected-card' : undefined}
                  >
                    exact_match={item.answer_exact_match} | context_recall={item.reference_context_recall}
                    {'\n'}
                    conflicts={item.conflicting_context_count ?? 0} | conflict_claims=
                    {item.conflict_claim_count ?? 0}
                    {'\n'}
                    insufficiency_claims={item.insufficiency_claim_count ?? 0}
                    {'\n'}
                    trace_id: {item.trace_id || 'n/a'}
                    {'\n'}
                    answer: {item.answer || 'n/a'}
                    {'\n'}
                    <button
                      type="button"
                      className="secondary"
                      onClick={() => setSelectedEvalResultIndex(reportIndex)}
                    >
                      聚焦样本
                    </button>
                    {diagnosisEnabled && currentEvalReportPath ? (
                      <button
                        type="button"
                        className="secondary"
                        onClick={() => {
                          setSelectedEvalResultIndex(reportIndex);
                          void diagnoseEvalResult(currentEvalReportPath, reportIndex, false);
                        }}
                      >
                        诊断
                      </button>
                    ) : null}
                    {item.trace_id ? (
                      <button
                        type="button"
                        className="secondary"
                        onClick={() => {
                          setSelectedTraceId(String(item.trace_id));
                          navigateToPage('investigate', 'traces-panel');
                        }}
                      >
                        打开 Trace
                      </button>
                    ) : null}
                    {item.query ? (
                      <button
                        type="button"
                        className="secondary"
                        onClick={() => {
                          prepareChatReplay({
                            query: item.query || '',
                            traceId: item.trace_id || undefined,
                            sourceLabel: `eval sample ${reportIndex}`,
                          });
                          navigateToPage('validate');
                        }}
                      >
                        回放到 Chat
                      </button>
                    ) : null}
                  </Card>
                ))
              ) : (
                <div className="empty-state">当前筛选下没有坏例。</div>
              )}
            </div>
          </div>

          {diagnosisEnabled ? (
            <>
              <StatusLine
                message={
                  diagnosisLoading
                    ? '正在分析坏例...'
                    : diagnosisError
                      ? diagnosisError
                      : diagnosisSummary || undefined
                }
                isError={!!diagnosisError}
              />
              {selectedEvalResultIndex !== null ? (
                <details className="details-panel">
                  <summary>查看当前坏例诊断</summary>
                  <JsonOutput data={diagnosis} placeholder="还没有诊断结果" />
                </details>
              ) : null}
            </>
          ) : null}
        </div>
      ) : evalEnabled ? (
        <div className="empty-state" style={{ marginTop: 18 }}>
          先运行一次 Eval。这里默认只展示核心指标和最值得先处理的坏例。
        </div>
      ) : null}

      {evalEnabled ? (
        <details className="details-panel" style={{ marginTop: 18 }}>
          <summary>高级筛选</summary>
          <div className="actions" style={{ marginTop: 12 }}>
            <button
              type="button"
              className={`secondary${!evalConflictOnly ? ' selected-card' : ''}`}
              onClick={() => setEvalConflictOnly(false)}
            >
              Eval 全部
            </button>
            <button
              type="button"
              className={`secondary${evalConflictOnly ? ' selected-card' : ''}`}
              onClick={() => setEvalConflictOnly(true)}
            >
              Eval 只看冲突
            </button>
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
              Eval 缺 conflict claim
            </button>
            <button
              type="button"
              className={`secondary${evalClaimFilter === 'insufficiency' ? ' selected-card' : ''}`}
              onClick={() => setEvalClaimFilter('insufficiency')}
            >
              Eval insufficiency
            </button>
          </div>
        </details>
      ) : null}

      {benchmarkEnabled ? (
        <details className="details-panel" style={{ marginTop: 18 }}>
          <summary>Benchmark（可选）</summary>
          <div className="stack" style={{ marginTop: 12 }}>
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
              <>
                <div className="metric-grid">
                  <div className="metric-card">
                    <span>Bad Cases</span>
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
                    <span>Latency P95</span>
                    <strong>{benchmarkAggregate.latency_seconds_p95 ?? 'n/a'}</strong>
                  </div>
                </div>

                {(benchmarkConflictOnly || benchmarkClaimFilter !== 'all') && (
                  <div className="status-line">
                    当前 Benchmark 筛选：{benchmarkConflictOnly ? '只看冲突' : '全部坏例'} |{' '}
                    {claimFilterLabel(benchmarkClaimFilter)}
                  </div>
                )}

                <div className="actions">
                  <button
                    type="button"
                    className={`secondary${!benchmarkConflictOnly ? ' selected-card' : ''}`}
                    onClick={() => setBenchmarkConflictOnly(false)}
                  >
                    Benchmark 全部
                  </button>
                  <button
                    type="button"
                    className={`secondary${benchmarkConflictOnly ? ' selected-card' : ''}`}
                    onClick={() => setBenchmarkConflictOnly(true)}
                  >
                    Benchmark 只看冲突
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
                    Benchmark 缺 conflict claim
                  </button>
                  <button
                    type="button"
                    className={`secondary${benchmarkClaimFilter === 'insufficiency' ? ' selected-card' : ''}`}
                    onClick={() => setBenchmarkClaimFilter('insufficiency')}
                  >
                    Benchmark insufficiency
                  </button>
                </div>

                <div className="cards">
                  {visibleBenchmarkBadCases.length ? (
                    visibleBenchmarkBadCases.slice(0, 4).map((item, index) => (
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
                        <button
                          type="button"
                          className="secondary"
                          onClick={() =>
                            setSelectedBenchmarkCaseKey(getBenchmarkCaseKey(item, index))
                          }
                        >
                          聚焦样本
                        </button>
                        {item.trace_id ? (
                          <button
                            type="button"
                            className="secondary"
                            onClick={() => {
                              setSelectedTraceId(String(item.trace_id));
                              navigateToPage('investigate', 'traces-panel');
                            }}
                          >
                            打开 Trace
                          </button>
                        ) : null}
                        {item.query ? (
                          <button
                            type="button"
                            className="secondary"
                            onClick={() => {
                              prepareChatReplay({
                                query: String(item.query || ''),
                                traceId: item.trace_id ? String(item.trace_id) : undefined,
                                sourceLabel: `benchmark case ${index + 1}`,
                              });
                              navigateToPage('validate');
                            }}
                          >
                            回放到 Chat
                          </button>
                        ) : null}
                      </Card>
                    ))
                  ) : (
                    <div className="empty-state">当前筛选下没有 benchmark 坏例。</div>
                  )}
                </div>

                {selectedBenchmarkCase ? (
                  <details className="details-panel">
                    <summary>查看当前 Benchmark 样本</summary>
                    <JsonOutput data={selectedBenchmarkCase} placeholder="还没有聚焦样本" />
                  </details>
                ) : null}
              </>
            ) : (
              <div className="empty-state">还没有 benchmark 结果。</div>
            )}
          </div>
        </details>
      ) : null}

      <details className="details-panel" style={{ marginTop: 18 }}>
        <summary>历史报告</summary>
        <div className="stack" style={{ marginTop: 12 }}>
          {evalEnabled ? (
            <>
              <StatusLine
                message={
                  evalReportsLoading
                    ? '正在加载评测报告...'
                    : evalReportLoading
                      ? '正在加载评测详情...'
                      : evalReportsError || evalReportError || undefined
                }
                isError={!!evalReportsError || !!evalReportError}
              />
              <div className="cards">
                {evalReports.length ? (
                  evalReports.slice(0, 6).map((report) => (
                    <button
                      key={report.path}
                      type="button"
                      className={`trace-item secondary${currentEvalReportPath === report.path ? ' selected-card' : ''}`}
                      onClick={() => loadEvalReport(report.path)}
                    >
                      <strong>{report.name}</strong>
                      <div className="mono">{report.path}</div>
                      <div className="muted">
                        exact_match={report.aggregate.answer_exact_match ?? 'n/a'} | conflict_hit_rate=
                        {formatRate(report.aggregate.conflicting_hit_rate)}
                      </div>
                      <div className="muted">updated={formatTimestamp(report.updated_at)}</div>
                    </button>
                  ))
                ) : (
                  <div className="empty-state">还没有评测历史。</div>
                )}
              </div>
            </>
          ) : null}

          {benchmarkEnabled ? (
            <>
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
              <div className="cards">
                {benchmarkReports.length ? (
                  benchmarkReports.slice(0, 6).map((report) => (
                    <button
                      key={report.path}
                      type="button"
                      className={`trace-item secondary${currentBenchmarkReportPath === report.path ? ' selected-card' : ''}`}
                      onClick={() => loadBenchmarkReport(report.path)}
                    >
                      <strong>{report.name}</strong>
                      <div className="mono">{report.path}</div>
                      <div className="muted">
                        bad_cases={report.aggregate.bad_case_count ?? 'n/a'} | latency_p95=
                        {report.aggregate.latency_seconds_p95 ?? 'n/a'}
                      </div>
                      <div className="muted">updated={formatTimestamp(report.updated_at)}</div>
                    </button>
                  ))
                ) : (
                  <div className="empty-state">还没有 benchmark 历史。</div>
                )}
              </div>
            </>
          ) : null}
        </div>
      </details>
    </Panel>
  );
}
