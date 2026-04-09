import { useEffect, useMemo } from 'react';
import { useAppStore } from '../stores/appStore';
import { benchmarkApi, evalApi, formatApiError } from '../api/client';
import { Panel, LoadingButton, StatusLine, Card } from '../components/common';

function formatRate(value: number | undefined): string {
  return value === undefined ? 'n/a' : `${(value * 100).toFixed(1)}%`;
}

function formatMetric(value: number | undefined): string {
  return value === undefined ? 'n/a' : String(value);
}

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

export function QualitySummaryPanel() {
  const {
    workspace,
    evalReports,
    evalReportsLoading,
    evalReportsError,
    loadEvalReports,
    loadEvalReport,
    benchmarkReports,
    benchmarkReportsLoading,
    benchmarkReportsError,
    loadBenchmarkReports,
    loadBenchmarkReport,
    setEvalConflictOnly,
    setBenchmarkConflictOnly,
    setEvalClaimFilter,
    setBenchmarkClaimFilter,
    setSelectedEvalResultIndex,
    setSelectedBenchmarkCaseKey,
    setSelectedTraceId,
    diagnoseEvalResult,
  } = useAppStore();

  const openEvalSample = async (options: {
    reportPath: string;
    conflictOnly: boolean;
    claimFilter: 'all' | 'missing_conflict' | 'insufficiency';
  }) => {
    setEvalConflictOnly(options.conflictOnly);
    setEvalClaimFilter(options.claimFilter);
    setSelectedEvalResultIndex(null);
    setSelectedTraceId('');
    try {
      const report = await evalApi.getReport(options.reportPath, workspace.apiKey);
      const match = report.results
        .map((item, reportIndex) => ({ item, reportIndex }))
        .find(({ item }) => {
          const isBadCase =
            Number(item.answer_exact_match ?? 1) < 1 || Number(item.reference_context_recall ?? 1) < 1;
          if (!isBadCase) {
            return false;
          }
          if (options.conflictOnly && Number(item.conflicting_context_count ?? 0) === 0) {
            return false;
          }
          if (options.claimFilter === 'missing_conflict') {
            return (
              Number(item.conflicting_context_count ?? 0) > 0 &&
              Number(item.conflict_claim_count ?? 0) === 0
            );
          }
          if (options.claimFilter === 'insufficiency') {
            return Number(item.insufficiency_claim_count ?? 0) > 0;
          }
          return true;
        });
      await loadEvalReport(options.reportPath);
      if (match) {
        setSelectedEvalResultIndex(match.reportIndex);
        setSelectedTraceId(String(match.item.trace_id || ''));
        await diagnoseEvalResult(options.reportPath, match.reportIndex, false);
      }
      openAdvancedPanel('eval-panel');
    } catch (error) {
      console.error(formatApiError(error));
      await loadEvalReport(options.reportPath);
      openAdvancedPanel('eval-panel');
    }
  };

  const openBenchmarkSample = async (options: {
    reportPath: string;
    conflictOnly: boolean;
    claimFilter: 'all' | 'missing_conflict' | 'insufficiency';
  }) => {
    setBenchmarkConflictOnly(options.conflictOnly);
    setBenchmarkClaimFilter(options.claimFilter);
    setSelectedBenchmarkCaseKey('');
    setSelectedTraceId('');
    try {
      const report = await benchmarkApi.getReport(options.reportPath, workspace.apiKey);
      const results = Array.isArray(report.results) ? (report.results as Record<string, unknown>[]) : [];
      const filtered = results.filter((item) => {
        const isBadCase =
          Number(item.answer_exact_match ?? 1) < 1 || Number(item.reference_context_recall ?? 1) < 1;
        if (!isBadCase) {
          return false;
        }
        if (options.conflictOnly && Number(item.conflicting_context_count ?? 0) === 0) {
          return false;
        }
        if (options.claimFilter === 'missing_conflict') {
          return (
            Number(item.conflicting_context_count ?? 0) > 0 &&
            Number(item.conflict_claim_count ?? 0) === 0
          );
        }
        if (options.claimFilter === 'insufficiency') {
          return Number(item.insufficiency_claim_count ?? 0) > 0;
        }
        return true;
      });
      const match =
        filtered.find(
          (item) =>
            typeof (item.diagnosis as Record<string, unknown> | undefined)?.summary === 'string' &&
            String((item.diagnosis as Record<string, unknown>).summary).trim().length > 0,
        ) ||
        filtered.find((item) => {
          if (options.conflictOnly && Number(item.conflicting_context_count ?? 0) === 0) {
            return false;
          }
          if (options.claimFilter === 'missing_conflict') {
            return (
              Number(item.conflicting_context_count ?? 0) > 0 &&
              Number(item.conflict_claim_count ?? 0) === 0
            );
          }
          if (options.claimFilter === 'insufficiency') {
            return Number(item.insufficiency_claim_count ?? 0) > 0;
          }
          return true;
        });
      await loadBenchmarkReport(options.reportPath);
      if (match) {
        setSelectedBenchmarkCaseKey(String(match.trace_id || match.query || 'benchmark-0'));
        setSelectedTraceId(String(match.trace_id || ''));
      }
      openAdvancedPanel('eval-panel');
    } catch (error) {
      console.error(formatApiError(error));
      await loadBenchmarkReport(options.reportPath);
      openAdvancedPanel('eval-panel');
    }
  };

  useEffect(() => {
    if (!evalReports.length) {
      loadEvalReports();
    }
    if (!benchmarkReports.length) {
      loadBenchmarkReports();
    }
  }, [
    benchmarkReports.length,
    evalReports.length,
    loadBenchmarkReports,
    loadEvalReports,
  ]);

  const latestEval = evalReports[0];
  const latestBenchmark = benchmarkReports[0];

  const qualitySignals = useMemo(
    () => ({
      evalConflictHitRate: latestEval?.aggregate.conflicting_hit_rate,
      evalConflictAvg: latestEval?.aggregate.conflicting_context_count_avg,
      evalExactMatch: latestEval?.aggregate.answer_exact_match,
      evalConflictClaimHitRate: latestEval?.aggregate.conflict_claim_hit_rate,
      evalInsufficiencyClaimHitRate: latestEval?.aggregate.insufficiency_claim_hit_rate,
      benchmarkConflictBadCases: latestBenchmark?.aggregate.conflicting_bad_case_count,
      benchmarkConflictHitRate: latestBenchmark?.aggregate.conflicting_hit_rate,
      benchmarkConflictClaimHitRate: latestBenchmark?.aggregate.conflict_claim_hit_rate,
      benchmarkInsufficiencyClaimHitRate: latestBenchmark?.aggregate.insufficiency_claim_hit_rate,
      benchmarkLatencyP95: latestBenchmark?.aggregate.latency_seconds_p95,
    }),
    [latestBenchmark, latestEval],
  );

  const evalConflictClaimGap = useMemo(() => {
    const hitRate = qualitySignals.evalConflictHitRate ?? 0;
    const claimHitRate = qualitySignals.evalConflictClaimHitRate ?? 0;
    return Math.max(hitRate - claimHitRate, 0);
  }, [qualitySignals.evalConflictClaimHitRate, qualitySignals.evalConflictHitRate]);

  const benchmarkConflictClaimGap = useMemo(() => {
    const hitRate = qualitySignals.benchmarkConflictHitRate ?? 0;
    const claimHitRate = qualitySignals.benchmarkConflictClaimHitRate ?? 0;
    return Math.max(hitRate - claimHitRate, 0);
  }, [
    qualitySignals.benchmarkConflictClaimHitRate,
    qualitySignals.benchmarkConflictHitRate,
  ]);

  return (
    <Panel
      title="离线质量概览"
      subtitle="汇总最近一次评测和 benchmark，判断冲突知识是否正在转化成坏例"
      actions={
        <>
          <LoadingButton loading={evalReportsLoading} onClick={loadEvalReports} variant="secondary">
            刷新评测
          </LoadingButton>
          <LoadingButton
            loading={benchmarkReportsLoading}
            onClick={loadBenchmarkReports}
            variant="secondary"
          >
            刷新 Benchmark
          </LoadingButton>
          <button
            type="button"
            className="secondary"
            onClick={() => {
              setEvalConflictOnly(false);
              setBenchmarkConflictOnly(false);
              setEvalClaimFilter('all');
              setBenchmarkClaimFilter('all');
              setSelectedEvalResultIndex(null);
              setSelectedBenchmarkCaseKey('');
              openAdvancedPanel('eval-panel');
            }}
          >
            查看 Eval
          </button>
        </>
      }
    >
      <StatusLine
        message={
          evalReportsLoading || benchmarkReportsLoading
            ? '正在加载离线质量报告...'
            : evalReportsError || benchmarkReportsError || undefined
        }
        isError={!!evalReportsError || !!benchmarkReportsError}
      />

      <div className="metric-grid">
        <div className="metric-card">
          <span>Eval Exact Match</span>
          <strong>{formatMetric(qualitySignals.evalExactMatch)}</strong>
        </div>
        <div className="metric-card">
          <span>Eval Conflict Hit Rate</span>
          <strong>{formatRate(qualitySignals.evalConflictHitRate)}</strong>
        </div>
        <div className="metric-card">
          <span>Eval Conflict Avg</span>
          <strong>{formatMetric(qualitySignals.evalConflictAvg)}</strong>
        </div>
        <div className="metric-card">
          <span>Eval Missing Conflict Claims</span>
          <strong>{formatRate(evalConflictClaimGap)}</strong>
          <button
            type="button"
            className="secondary"
            onClick={() =>
              latestEval &&
              void openEvalSample({
                reportPath: latestEval.path,
                conflictOnly: true,
                claimFilter: 'missing_conflict',
              })
            }
            style={{ marginTop: 8 }}
          >
            查看缺口
          </button>
        </div>
        <div className="metric-card">
          <span>Eval Insufficiency Claims</span>
          <strong>{formatRate(qualitySignals.evalInsufficiencyClaimHitRate)}</strong>
          <button
            type="button"
            className="secondary"
            onClick={() =>
              latestEval &&
              void openEvalSample({
                reportPath: latestEval.path,
                conflictOnly: false,
                claimFilter: 'insufficiency',
              })
            }
            style={{ marginTop: 8 }}
          >
            查看不足
          </button>
        </div>
        <div className="metric-card">
          <span>Benchmark Conflict Bad Cases</span>
          <strong>{formatMetric(qualitySignals.benchmarkConflictBadCases)}</strong>
          <button
            type="button"
            className="secondary"
            onClick={() =>
              latestBenchmark &&
              void openBenchmarkSample({
                reportPath: latestBenchmark.path,
                conflictOnly: true,
                claimFilter: 'all',
              })
            }
            style={{ marginTop: 8 }}
          >
            查看坏例
          </button>
        </div>
        <div className="metric-card">
          <span>Benchmark Conflict Hit Rate</span>
          <strong>{formatRate(qualitySignals.benchmarkConflictHitRate)}</strong>
        </div>
        <div className="metric-card">
          <span>Benchmark Missing Conflict Claims</span>
          <strong>{formatRate(benchmarkConflictClaimGap)}</strong>
          <button
            type="button"
            className="secondary"
            onClick={() =>
              latestBenchmark &&
              void openBenchmarkSample({
                reportPath: latestBenchmark.path,
                conflictOnly: true,
                claimFilter: 'missing_conflict',
              })
            }
            style={{ marginTop: 8 }}
          >
            查看缺口
          </button>
        </div>
        <div className="metric-card">
          <span>Benchmark Insufficiency Claims</span>
          <strong>{formatRate(qualitySignals.benchmarkInsufficiencyClaimHitRate)}</strong>
          <button
            type="button"
            className="secondary"
            onClick={() =>
              latestBenchmark &&
              void openBenchmarkSample({
                reportPath: latestBenchmark.path,
                conflictOnly: false,
                claimFilter: 'insufficiency',
              })
            }
            style={{ marginTop: 8 }}
          >
            查看不足
          </button>
        </div>
        <div className="metric-card">
          <span>Benchmark Latency P95</span>
          <strong>{formatMetric(qualitySignals.benchmarkLatencyP95)}</strong>
        </div>
      </div>

      <div style={{ marginTop: 16 }} className="cards">
        <Card title="最近评测">
          {latestEval ? (
            <>
              {latestEval.name}
              {'\n'}
              records={latestEval.records} | conflict_hit_rate=
              {formatRate(latestEval.aggregate.conflicting_hit_rate)}
              {'\n'}
              conflict_claim_hit_rate={formatRate(latestEval.aggregate.conflict_claim_hit_rate)}
              {'\n'}
              <button
                type="button"
                className="secondary"
                onClick={() =>
                  void openEvalSample({
                    reportPath: latestEval.path,
                    conflictOnly: true,
                    claimFilter: 'missing_conflict',
                  })
                }
              >
                打开缺口视图
              </button>
            </>
          ) : (
            '还没有评测报告。'
          )}
        </Card>
        <Card title="最近 Benchmark">
          {latestBenchmark ? (
            <>
              {latestBenchmark.name}
              {'\n'}
              bad_cases={latestBenchmark.aggregate.bad_case_count ?? 'n/a'} | conflict_bad_cases=
              {latestBenchmark.aggregate.conflicting_bad_case_count ?? 'n/a'}
              {'\n'}
              conflict_claim_hit_rate={formatRate(latestBenchmark.aggregate.conflict_claim_hit_rate)}
              {'\n'}
              <button
                type="button"
                className="secondary"
                onClick={() =>
                  void openBenchmarkSample({
                    reportPath: latestBenchmark.path,
                    conflictOnly: true,
                    claimFilter: 'missing_conflict',
                  })
                }
              >
                打开缺口视图
              </button>
            </>
          ) : (
            '还没有 benchmark 报告。'
          )}
        </Card>
      </div>
    </Panel>
  );
}
