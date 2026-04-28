import axios, { AxiosError } from 'axios';

import type {
  HealthResponse,
  IngestRequest,
  IngestResponse,
  DocumentSummary,
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
  const detail = parsed.detail || parsed.message;
  const normalized = detail.toLowerCase();
  if (normalized.includes('quota exceeded') || normalized.includes('resource_exhausted')) {
    return '当前模型服务配额已用完，请稍后再试，或切换到可用的模型配置。';
  }
  if (normalized.includes('api key not valid') || normalized.includes('invalid or missing api key')) {
    return '当前模型或业务接口的 API Key 无效，请先检查配置。';
  }
  if (normalized.includes('rag_api_keys not configured')) {
    return '后端未配置 RAG_API_KEYS。生产或共享环境请配置业务 key；本地开发可显式设置 RAG_AUTH_DISABLED=true。';
  }
  if (normalized.includes('unsupported file type')) {
    return parsed.detail || '文件类型不受支持，请上传 PDF、Markdown、TXT、HTML 或常见图片。';
  }
  if (normalized.includes('evaluation is disabled') || normalized.includes('benchmark is disabled because evaluation is off')) {
    return '当前实例没有开启离线评测能力；如果需要 Eval 或 Benchmark，请在后端启用 RAG_EVAL_ENABLED=true。';
  }
  if (normalized.includes('diagnosis is disabled') || normalized.includes('benchmark is disabled because diagnosis is off')) {
    return '当前实例没有开启诊断能力；如果需要自动诊断或完整 Benchmark，请在后端启用 RAG_DIAGNOSIS_ENABLED=true。';
  }
  if (normalized.includes('returned empty content') || normalized.includes('produced no chunks')) {
    return '文件已经上传，但模型没有成功读出可检索内容。请优先尝试更清晰的 PDF，或改用支持文档理解的模型配置。';
  }
  if (parsed.detail && parsed.detail !== parsed.message) {
    return `${parsed.message}: ${parsed.detail}`;
  }
  return parsed.detail || parsed.message;
}

export const healthApi = {
  get: (apiKey?: string) =>
    api
      .get<HealthResponse>('/health/detail', {
        headers: authHeaders(apiKey),
      })
      .then((r) => r.data),
};

export const businessIngestApi = {
  run: (payload: IngestRequest, apiKey?: string) =>
    api
      .post<IngestResponse>('/v1/rag/ingest', payload, {
        headers: authHeaders(apiKey),
      })
      .then((r) => r.data),
  upload: (
    payload: {
      files: File[];
      kb_id?: string;
      tenant_id?: string;
    },
    apiKey?: string,
  ) => {
    const form = new FormData();
    payload.files.forEach((file) => form.append('files', file));
    if (payload.kb_id) {
      form.append('kb_id', payload.kb_id);
    }
    if (payload.tenant_id) {
      form.append('tenant_id', payload.tenant_id);
    }
    return axios
      .post<IngestResponse>(`${API_BASE}/v1/rag/ingest/upload`, form, {
        headers: authHeaders(apiKey),
      })
      .then((r) => r.data);
  },
  listDocuments: (
    payload: {
      kb_id: string;
      tenant_id?: string;
    },
    apiKey?: string,
  ) =>
    api
      .get<DocumentSummary[]>('/v1/rag/documents', {
        params: payload,
        headers: authHeaders(apiKey),
      })
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
