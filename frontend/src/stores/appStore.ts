import { create } from 'zustand';
import type {
  HealthResponse,
  ChatResponse,
  RetrievalDebugResponse,
  TraceSummary,
  TraceRecord,
  IngestResponse,
  DocumentSummary,
  EvalRunResponse,
  EvalDatasetSummary,
  EvalReportSummary,
  EvalReportDetail,
  DiagnosisResponse,
  BenchmarkRunResponse,
  BenchmarkReportSummary,
  WorkspaceSettings,
} from '../types';
import {
  healthApi,
  businessIngestApi,
  businessChatApi,
  businessFeedbackApi,
  debugApi,
  traceApi,
  evalApi,
  benchmarkApi,
  diagnosisApi,
  formatApiError,
} from '../api/client';

const WORKSPACE_STORAGE_KEY = 'nano-rag-workspace';

function loadInitialWorkspace(): WorkspaceSettings {
  if (typeof window === 'undefined') {
    return {
      kbId: 'default',
      tenantId: 'demo-tenant',
      sessionId: 'session-web',
      apiKey: '',
    };
  }

  try {
    const raw = window.localStorage.getItem(WORKSPACE_STORAGE_KEY);
    if (!raw) {
      return {
        kbId: 'default',
        tenantId: 'demo-tenant',
        sessionId: 'session-web',
        apiKey: '',
      };
    }
    const parsed = JSON.parse(raw) as Partial<WorkspaceSettings>;
    return {
      kbId: parsed.kbId?.trim() || 'default',
      tenantId: parsed.tenantId?.trim() || 'demo-tenant',
      sessionId: parsed.sessionId?.trim() || 'session-web',
      apiKey: '',
    };
  } catch {
    return {
      kbId: 'default',
      tenantId: 'demo-tenant',
      sessionId: 'session-web',
      apiKey: '',
    };
  }
}

function persistWorkspace(workspace: WorkspaceSettings): void {
  if (typeof window === 'undefined') {
    return;
  }
  window.localStorage.setItem(
    WORKSPACE_STORAGE_KEY,
    JSON.stringify({
      kbId: workspace.kbId,
      tenantId: workspace.tenantId,
      sessionId: workspace.sessionId,
    }),
  );
}

export type TraceClaimFilter = 'all' | 'missing_conflict' | 'insufficiency' | 'conditional';
export type EvalClaimFilter = 'all' | 'missing_conflict' | 'insufficiency';
export interface ChatReplayDraft {
  query: string;
  kbId?: string;
  tenantId?: string;
  sessionId?: string;
  traceId?: string;
  topK?: number;
  sourceLabel?: string;
}

interface AppState {
  workspace: WorkspaceSettings;
  updateWorkspace: (next: Partial<WorkspaceSettings>) => void;
  traceConflictOnly: boolean;
  evalConflictOnly: boolean;
  benchmarkConflictOnly: boolean;
  traceClaimFilter: TraceClaimFilter;
  evalClaimFilter: EvalClaimFilter;
  benchmarkClaimFilter: EvalClaimFilter;
  setTraceConflictOnly: (value: boolean) => void;
  setEvalConflictOnly: (value: boolean) => void;
  setBenchmarkConflictOnly: (value: boolean) => void;
  setTraceClaimFilter: (value: TraceClaimFilter) => void;
  setEvalClaimFilter: (value: EvalClaimFilter) => void;
  setBenchmarkClaimFilter: (value: EvalClaimFilter) => void;
  clearAdvancedFilters: () => void;
  chatReplayDraft: ChatReplayDraft | null;
  prepareChatReplay: (draft: ChatReplayDraft) => void;
  clearChatReplay: () => void;

  health: HealthResponse | null;
  healthLoading: boolean;
  healthError: string | null;
  loadHealth: () => Promise<void>;

  ingestResult: IngestResponse | null;
  ingestLoading: boolean;
  ingestError: string | null;
  runIngest: (path: string) => Promise<void>;
  runIngestUpload: (files: File[]) => Promise<void>;

  documents: DocumentSummary[];
  documentsLoading: boolean;
  documentsError: string | null;
  loadDocuments: () => Promise<void>;

  chatResult: ChatResponse | null;
  chatLoading: boolean;
  chatError: string | null;
  sendChat: (query: string, topK?: number) => Promise<void>;

  feedbackResult: { status: string; feedback_id: string } | null;
  feedbackLoading: boolean;
  feedbackError: string | null;
  submitFeedback: (rating: 'up' | 'down', comment?: string) => Promise<void>;

  debugResult: RetrievalDebugResponse | null;
  debugLoading: boolean;
  debugError: string | null;
  runDebug: (query: string, topK?: number) => Promise<void>;

  traces: TraceSummary[];
  tracesLoading: boolean;
  tracesError: string | null;
  loadTraces: () => Promise<void>;

  currentTrace: TraceRecord | null;
  traceLoading: boolean;
  traceError: string | null;
  loadTrace: (traceId: string) => Promise<void>;

  evalResult: EvalRunResponse | null;
  evalLoading: boolean;
  evalError: string | null;
  evalDatasets: EvalDatasetSummary[];
  evalDatasetsLoading: boolean;
  evalDatasetsError: string | null;
  loadEvalDatasets: () => Promise<void>;
  evalReports: EvalReportSummary[];
  evalReportsLoading: boolean;
  evalReportsError: string | null;
  loadEvalReports: () => Promise<void>;
  currentEvalReport: EvalReportDetail | null;
  currentEvalReportPath: string;
  selectedEvalResultIndex: number | null;
  evalReportLoading: boolean;
  evalReportError: string | null;
  loadEvalReport: (path: string) => Promise<void>;
  setSelectedEvalResultIndex: (value: number | null) => void;
  runEval: (datasetPath: string, outputPath?: string) => Promise<void>;

  benchmarkResult: BenchmarkRunResponse | null;
  benchmarkLoading: boolean;
  benchmarkError: string | null;
  benchmarkReports: BenchmarkReportSummary[];
  benchmarkReportsLoading: boolean;
  benchmarkReportsError: string | null;
  currentBenchmarkReport: Record<string, unknown> | null;
  currentBenchmarkReportPath: string;
  selectedBenchmarkCaseKey: string;
  benchmarkReportLoading: boolean;
  benchmarkReportError: string | null;
  loadBenchmarkReports: () => Promise<void>;
  loadBenchmarkReport: (path: string) => Promise<void>;
  setSelectedBenchmarkCaseKey: (value: string) => void;
  runBenchmark: (datasetPath: string, outputPath?: string) => Promise<void>;

  diagnosis: DiagnosisResponse | null;
  diagnosisLoading: boolean;
  diagnosisError: string | null;
  diagnoseAuto: (includeAi?: boolean) => Promise<void>;
  diagnoseTrace: (traceId: string, includeAi?: boolean) => Promise<void>;
  diagnoseEvalResult: (reportPath: string, resultIndex: number, includeAi?: boolean) => Promise<void>;

  selectedTraceId: string;
  setSelectedTraceId: (id: string) => void;
}

export const useAppStore = create<AppState>((set, get) => ({
  workspace: loadInitialWorkspace(),
  updateWorkspace: (next) => {
    const workspace = { ...get().workspace, ...next };
    persistWorkspace(workspace);
    set({ workspace });
  },
  traceConflictOnly: false,
  evalConflictOnly: false,
  benchmarkConflictOnly: false,
  traceClaimFilter: 'all',
  evalClaimFilter: 'all',
  benchmarkClaimFilter: 'all',
  setTraceConflictOnly: (value) => set({ traceConflictOnly: value }),
  setEvalConflictOnly: (value) => set({ evalConflictOnly: value }),
  setBenchmarkConflictOnly: (value) => set({ benchmarkConflictOnly: value }),
  setTraceClaimFilter: (value) => set({ traceClaimFilter: value }),
  setEvalClaimFilter: (value) => set({ evalClaimFilter: value }),
  setBenchmarkClaimFilter: (value) => set({ benchmarkClaimFilter: value }),
  clearAdvancedFilters: () =>
    set({
      traceConflictOnly: false,
      evalConflictOnly: false,
      benchmarkConflictOnly: false,
      traceClaimFilter: 'all',
      evalClaimFilter: 'all',
      benchmarkClaimFilter: 'all',
      selectedEvalResultIndex: null,
      selectedBenchmarkCaseKey: '',
    }),
  chatReplayDraft: null,
  prepareChatReplay: (draft) => set({ chatReplayDraft: draft }),
  clearChatReplay: () => set({ chatReplayDraft: null }),

  health: null,
  healthLoading: false,
  healthError: null,
  loadHealth: async () => {
    set({ healthLoading: true, healthError: null });
    try {
      const data = await healthApi.get();
      set({ health: data, healthLoading: false });
    } catch (e) {
      set({ healthError: formatApiError(e), healthLoading: false });
    }
  },

  ingestResult: null,
  ingestLoading: false,
  ingestError: null,
  runIngest: async (path: string) => {
    set({ ingestLoading: true, ingestError: null });
    try {
      const { workspace } = get();
      const data = await businessIngestApi.run(
        {
          path,
          kb_id: workspace.kbId,
          tenant_id: workspace.tenantId || undefined,
        },
        workspace.apiKey,
      );
      set({ ingestResult: data, ingestLoading: false });
      await get().loadHealth();
      await get().loadDocuments();
    } catch (e) {
      set({ ingestError: formatApiError(e), ingestLoading: false });
    }
  },
  runIngestUpload: async (files: File[]) => {
    set({ ingestLoading: true, ingestError: null });
    try {
      const { workspace } = get();
      const data = await businessIngestApi.upload(
        {
          files,
          kb_id: workspace.kbId,
          tenant_id: workspace.tenantId || undefined,
        },
        workspace.apiKey,
      );
      set({ ingestResult: data, ingestLoading: false });
      await get().loadHealth();
      await get().loadDocuments();
    } catch (e) {
      set({ ingestError: formatApiError(e), ingestLoading: false });
    }
  },

  documents: [],
  documentsLoading: false,
  documentsError: null,
  loadDocuments: async () => {
    set({ documentsLoading: true, documentsError: null });
    try {
      const { workspace } = get();
      const data = await businessIngestApi.listDocuments(
        {
          kb_id: workspace.kbId,
          tenant_id: workspace.tenantId || undefined,
        },
        workspace.apiKey,
      );
      set({ documents: data, documentsLoading: false });
    } catch (e) {
      set({ documentsError: formatApiError(e), documentsLoading: false });
    }
  },

  chatResult: null,
  chatLoading: false,
  chatError: null,
  sendChat: async (query: string, topK?: number) => {
    set({ chatLoading: true, chatError: null, feedbackResult: null, feedbackError: null });
    try {
      const { workspace } = get();
      const data = await businessChatApi.send(
        {
          query,
          top_k: topK,
          kb_id: workspace.kbId,
          tenant_id: workspace.tenantId || undefined,
          session_id: workspace.sessionId || undefined,
        },
        workspace.apiKey,
      );
      set({ chatResult: data, chatLoading: false });
      if (data.trace_id) {
        set({ selectedTraceId: data.trace_id });
      }
      await get().loadTraces();
    } catch (e) {
      set({ chatError: formatApiError(e), chatLoading: false });
    }
  },

  feedbackResult: null,
  feedbackLoading: false,
  feedbackError: null,
  submitFeedback: async (rating, comment) => {
    const traceId = get().chatResult?.trace_id;
    if (!traceId) {
      set({ feedbackError: '当前没有可提交反馈的 trace。请先完成一次问答。' });
      return;
    }
    set({ feedbackLoading: true, feedbackError: null, feedbackResult: null });
    try {
      const { workspace } = get();
      const data = await businessFeedbackApi.submit(
        {
          trace_id: traceId,
          rating,
          kb_id: workspace.kbId,
          tenant_id: workspace.tenantId || undefined,
          session_id: workspace.sessionId || undefined,
          comment: comment?.trim() || undefined,
          tags: ['frontend'],
        },
        workspace.apiKey,
      );
      set({ feedbackResult: data, feedbackLoading: false });
    } catch (e) {
      set({ feedbackError: formatApiError(e), feedbackLoading: false });
    }
  },

  debugResult: null,
  debugLoading: false,
  debugError: null,
  runDebug: async (query: string, topK?: number) => {
    set({ debugLoading: true, debugError: null });
    try {
      const { workspace } = get();
      const data = await debugApi.retrieve({
        query,
        top_k: topK,
        kb_id: workspace.kbId,
        tenant_id: workspace.tenantId || undefined,
        session_id: workspace.sessionId || undefined,
      }, workspace.apiKey);
      set({ debugResult: data, debugLoading: false });
      if (data.trace_id) {
        set({ selectedTraceId: data.trace_id });
      }
      await get().loadTraces();
    } catch (e) {
      set({ debugError: formatApiError(e), debugLoading: false });
    }
  },

  traces: [],
  tracesLoading: false,
  tracesError: null,
  loadTraces: async () => {
    set({ tracesLoading: true, tracesError: null });
    try {
      const { workspace } = get();
      const data = await traceApi.list(workspace.apiKey);
      set({ traces: data.items || [], tracesLoading: false });
    } catch (e) {
      set({ tracesError: formatApiError(e), tracesLoading: false });
    }
  },

  currentTrace: null,
  traceLoading: false,
  traceError: null,
  loadTrace: async (traceId: string) => {
    set({ traceLoading: true, traceError: null });
    try {
      const { workspace } = get();
      const data = await traceApi.get(traceId, workspace.apiKey);
      set({ currentTrace: data, traceLoading: false });
    } catch (e) {
      set({ traceError: formatApiError(e), traceLoading: false });
    }
  },

  evalResult: null,
  evalLoading: false,
  evalError: null,
  evalDatasets: [],
  evalDatasetsLoading: false,
  evalDatasetsError: null,
  loadEvalDatasets: async () => {
    set({ evalDatasetsLoading: true, evalDatasetsError: null });
    try {
      const { workspace } = get();
      const data = await evalApi.listDatasets(workspace.apiKey);
      set({ evalDatasets: data, evalDatasetsLoading: false });
    } catch (e) {
      set({ evalDatasetsError: formatApiError(e), evalDatasetsLoading: false });
    }
  },
  evalReports: [],
  evalReportsLoading: false,
  evalReportsError: null,
  loadEvalReports: async () => {
    set({ evalReportsLoading: true, evalReportsError: null });
    try {
      const { workspace } = get();
      const data = await evalApi.listReports(workspace.apiKey);
      set({ evalReports: data, evalReportsLoading: false });
    } catch (e) {
      set({ evalReportsError: formatApiError(e), evalReportsLoading: false });
    }
  },
  currentEvalReport: null,
  currentEvalReportPath: '',
  selectedEvalResultIndex: null,
  evalReportLoading: false,
  evalReportError: null,
  loadEvalReport: async (path: string) => {
    set({ evalReportLoading: true, evalReportError: null });
    try {
      const { workspace } = get();
      const data = await evalApi.getReport(path, workspace.apiKey);
      set({
        currentEvalReport: data,
        currentEvalReportPath: path,
        selectedEvalResultIndex: null,
        evalReportLoading: false,
      });
    } catch (e) {
      set({ evalReportError: formatApiError(e), evalReportLoading: false });
    }
  },
  runEval: async (datasetPath: string, outputPath?: string) => {
    set({ evalLoading: true, evalError: null });
    try {
      const { workspace } = get();
      const data = await evalApi.run(datasetPath, outputPath, workspace.apiKey);
      set({
        evalResult: data,
        currentEvalReport: data.report,
        currentEvalReportPath: data.output_path || '',
        selectedEvalResultIndex: null,
        evalLoading: false,
      });
      await get().loadEvalReports();
    } catch (e) {
      set({ evalError: formatApiError(e), evalLoading: false });
    }
  },

  setSelectedEvalResultIndex: (value) => set({ selectedEvalResultIndex: value }),

  benchmarkResult: null,
  benchmarkLoading: false,
  benchmarkError: null,
  benchmarkReports: [],
  benchmarkReportsLoading: false,
  benchmarkReportsError: null,
  currentBenchmarkReport: null,
  currentBenchmarkReportPath: '',
  selectedBenchmarkCaseKey: '',
  benchmarkReportLoading: false,
  benchmarkReportError: null,
  loadBenchmarkReports: async () => {
    set({ benchmarkReportsLoading: true, benchmarkReportsError: null });
    try {
      const { workspace } = get();
      const data = await benchmarkApi.listReports(workspace.apiKey);
      set({ benchmarkReports: data, benchmarkReportsLoading: false });
    } catch (e) {
      set({ benchmarkReportsError: formatApiError(e), benchmarkReportsLoading: false });
    }
  },
  loadBenchmarkReport: async (path: string) => {
    set({ benchmarkReportLoading: true, benchmarkReportError: null });
    try {
      const { workspace } = get();
      const data = await benchmarkApi.getReport(path, workspace.apiKey);
      set({
        currentBenchmarkReport: data,
        currentBenchmarkReportPath: path,
        selectedBenchmarkCaseKey: '',
        benchmarkReportLoading: false,
      });
    } catch (e) {
      set({ benchmarkReportError: formatApiError(e), benchmarkReportLoading: false });
    }
  },
  runBenchmark: async (datasetPath: string, outputPath?: string) => {
    set({ benchmarkLoading: true, benchmarkError: null });
    try {
      const { workspace } = get();
      const data = await benchmarkApi.run(datasetPath, outputPath, workspace.apiKey);
      set({
        benchmarkResult: data,
        currentBenchmarkReport: data.report,
        currentBenchmarkReportPath: data.output_path || '',
        selectedBenchmarkCaseKey: '',
        benchmarkLoading: false,
      });
      await get().loadBenchmarkReports();
    } catch (e) {
      set({ benchmarkError: formatApiError(e), benchmarkLoading: false });
    }
  },
  setSelectedBenchmarkCaseKey: (value) => set({ selectedBenchmarkCaseKey: value }),

  diagnosis: null,
  diagnosisLoading: false,
  diagnosisError: null,
  diagnoseAuto: async (includeAi = true) => {
    set({ diagnosisLoading: true, diagnosisError: null });
    try {
      const { workspace } = get();
      const data = await diagnosisApi.auto(includeAi, workspace.apiKey);
      set({ diagnosis: data, diagnosisLoading: false });
    } catch (e) {
      set({ diagnosisError: formatApiError(e), diagnosisLoading: false });
    }
  },
  diagnoseTrace: async (traceId: string, includeAi = false) => {
    set({ diagnosisLoading: true, diagnosisError: null });
    try {
      const { workspace } = get();
      const data = await diagnosisApi.trace(traceId, includeAi, workspace.apiKey);
      set({ diagnosis: data, diagnosisLoading: false });
    } catch (e) {
      set({ diagnosisError: formatApiError(e), diagnosisLoading: false });
    }
  },
  diagnoseEvalResult: async (reportPath: string, resultIndex: number, includeAi = false) => {
    set({ diagnosisLoading: true, diagnosisError: null });
    try {
      const { workspace } = get();
      const data = await diagnosisApi.eval(reportPath, resultIndex, includeAi, workspace.apiKey);
      set({ diagnosis: data, diagnosisLoading: false });
    } catch (e) {
      set({ diagnosisError: formatApiError(e), diagnosisLoading: false });
    }
  },

  selectedTraceId: '',
  setSelectedTraceId: (id: string) => set({ selectedTraceId: id }),
}));
