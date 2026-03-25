import axios, { AxiosError } from 'axios';

import type {
  HealthResponse,
  IngestRequest,
  IngestResponse,
  ChatRequest,
  ChatResponse,
  RetrievalDebugResponse,
  PaginatedTracesResponse,
  TraceRecord,
  EvalDatasetSummary,
  EvalReportSummary,
  EvalReportDetail,
  EvalRunResponse,
  BenchmarkRunResponse,
  BenchmarkReportSummary,
  DiagnosisResponse,
} from '../types';

const API_BASE = import.meta.env.VITE_API_BASE || '';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

function authHeaders(apiKey?: string) {
  if (!apiKey?.trim()) {
    return undefined;
  }
  return {
    'X-API-Key': apiKey.trim(),
  };
}

export interface ApiError {
  message: string;
  detail?: string;
}

export function parseApiError(error: unknown): ApiError {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<{ detail?: string }>;
    return {
      message: axiosError.message,
      detail: axiosError.response?.data?.detail,
    };
  }
  if (error instanceof Error) {
    return { message: error.message };
  }
  return { message: 'Unknown error' };
}

export function formatApiError(error: unknown): string {
  const parsed = parseApiError(error);
  if (parsed.detail && parsed.detail !== parsed.message) {
    return `${parsed.message}: ${parsed.detail}`;
  }
  return parsed.detail || parsed.message;
}

export const healthApi = {
  get: () => api.get<HealthResponse>('/health').then((r) => r.data),
};

export const ingestApi = {
  run: (path: string) =>
    api.post<IngestResponse>('/ingest', { path }).then((r) => r.data),
};

export const businessIngestApi = {
  run: (payload: IngestRequest, apiKey?: string) =>
    api
      .post<IngestResponse>('/v1/rag/ingest', payload, {
        headers: authHeaders(apiKey),
      })
      .then((r) => r.data),
};

export const chatApi = {
  send: (query: string, topK?: number) =>
    api
      .post<ChatResponse>('/chat', { query, top_k: topK })
      .then((r) => r.data),
};

export const businessChatApi = {
  send: (payload: ChatRequest, apiKey?: string) =>
    api
      .post<ChatResponse>('/v1/rag/chat', payload, {
        headers: authHeaders(apiKey),
      })
      .then((r) => r.data),
};

export const businessFeedbackApi = {
  submit: (
    payload: {
      trace_id: string;
      rating: string;
      kb_id: string;
      tenant_id?: string;
      session_id?: string;
      comment?: string;
      tags?: string[];
    },
    apiKey?: string,
  ) =>
    api
      .post<{ status: string; feedback_id: string }>('/v1/rag/feedback', payload, {
        headers: authHeaders(apiKey),
      })
      .then((r) => r.data),
};

export const debugApi = {
  retrieve: (payload: ChatRequest, apiKey?: string) =>
    api
      .post<RetrievalDebugResponse>('/retrieve/debug', payload, {
        headers: authHeaders(apiKey),
      })
      .then((r) => r.data),
};

export const traceApi = {
  list: (apiKey?: string) =>
    api
      .get<PaginatedTracesResponse>('/traces', { headers: authHeaders(apiKey) })
      .then((r) => r.data),
  get: (traceId: string, apiKey?: string) =>
    api
      .get<TraceRecord>(`/traces/${traceId}`, { headers: authHeaders(apiKey) })
      .then((r) => r.data),
};

export const evalApi = {
  listDatasets: (apiKey?: string) =>
    api
      .get<EvalDatasetSummary[]>('/eval/datasets', { headers: authHeaders(apiKey) })
      .then((r) => r.data),
  listReports: (apiKey?: string) =>
    api
      .get<EvalReportSummary[]>('/eval/reports', { headers: authHeaders(apiKey) })
      .then((r) => r.data),
  getReport: (path: string, apiKey?: string) =>
    api
      .get<EvalReportDetail>('/eval/reports/detail', {
        params: { path },
        headers: authHeaders(apiKey),
      })
      .then((r) => r.data),
  run: (datasetPath: string, outputPath?: string, apiKey?: string) =>
    api
      .post<EvalRunResponse>(
        '/eval/run',
        {
          dataset_path: datasetPath,
          output_path: outputPath,
        },
        {
          headers: authHeaders(apiKey),
        },
      )
      .then((r) => r.data),
};

export const benchmarkApi = {
  listReports: (apiKey?: string) =>
    api
      .get<BenchmarkReportSummary[]>('/benchmark/reports', { headers: authHeaders(apiKey) })
      .then((r) => r.data),
  getReport: (path: string, apiKey?: string) =>
    api
      .get<Record<string, unknown>>('/benchmark/reports/detail', {
        params: { path },
        headers: authHeaders(apiKey),
      })
      .then((r) => r.data),
  run: (datasetPath: string, outputPath?: string, apiKey?: string) =>
    api
      .post<BenchmarkRunResponse>(
        '/v1/rag/benchmark/run',
        {
          dataset_path: datasetPath,
          output_path: outputPath,
        },
        {
          headers: authHeaders(apiKey),
        },
      )
      .then((r) => r.data),
};

export const diagnosisApi = {
  auto: (includeAi = true, apiKey?: string) =>
    api
      .post<DiagnosisResponse>(
        '/diagnose/auto',
        {
          include_ai: includeAi,
        },
        {
          headers: authHeaders(apiKey),
        },
      )
      .then((r) => r.data),
  trace: (traceId: string, includeAi = false, apiKey?: string) =>
    api
      .post<DiagnosisResponse>(
        '/diagnose/trace',
        {
          trace_id: traceId,
          include_ai: includeAi,
        },
        {
          headers: authHeaders(apiKey),
        },
      )
      .then((r) => r.data),
  eval: (reportPath: string, resultIndex: number, includeAi = false, apiKey?: string) =>
    api
      .post<DiagnosisResponse>(
        '/diagnose/eval',
        {
          report_path: reportPath,
          result_index: resultIndex,
          include_ai: includeAi,
        },
        {
          headers: authHeaders(apiKey),
        },
      )
      .then((r) => r.data),
};
