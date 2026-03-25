export interface HealthResponse {
  status: string;
  service: string;
  vectorstore_backend: string;
  gateway_mode: string;
  gateway: {
    base_url: string;
    reachable: boolean;
    error: string | null;
  };
  phoenix: {
    collector_endpoint: string;
    ui_endpoint: string;
    reachable: boolean;
    error: string | null;
  };
  vectorstore: {
    status: string;
    error: string | null;
    details: Record<string, unknown>;
  };
  parsed_dir: string;
  trace_count: number;
}

export interface IngestRequest {
  path: string;
  kb_id?: string;
  tenant_id?: string;
}

export interface IngestResponse {
  status?: string;
  kb_id?: string;
  tenant_id?: string | null;
  documents: number;
  chunks: number;
}

export interface ChatRequest {
  query: string;
  top_k?: number;
  kb_id?: string;
  tenant_id?: string;
  session_id?: string;
}

export interface Citation {
  chunk_id: string;
  source: string;
  score?: number;
}

export interface ChatResponse {
  answer: string;
  citations: Citation[];
  contexts: Record<string, unknown>[];
  trace_id?: string;
  kb_id?: string;
  tenant_id?: string | null;
  session_id?: string | null;
}

export interface RetrievalDebugResponse {
  query: string;
  retrieved: Record<string, unknown>[];
  reranked: Record<string, unknown>[];
  contexts: Record<string, unknown>[];
  trace_id?: string;
}

export interface TraceSummary {
  trace_id: string;
  latency_seconds?: number;
  query?: string;
  model_alias?: string;
  prompt_version?: string;
  context_count?: number;
  kb_id?: string | null;
  tenant_id?: string | null;
  session_id?: string | null;
}

export interface TraceRecord {
  trace_id: string;
  latency_seconds?: number;
  query?: string;
  sample_id?: string;
  kb_id?: string | null;
  tenant_id?: string | null;
  session_id?: string | null;
  retrieved_chunk_ids: string[];
  reranked_chunk_ids: string[];
  retrieved: Record<string, unknown>[];
  reranked: Record<string, unknown>[];
  contexts: Record<string, unknown>[];
  citations: Record<string, unknown>[];
  answer?: string;
  model_alias?: string;
  embedding_model_alias?: string;
  rerank_model_alias?: string;
  prompt_version?: string;
  prompt_messages: Record<string, unknown>[];
  generation_finish_reason?: string;
  generation_usage: Record<string, unknown>;
  retrieval_params: Record<string, unknown>;
  step_latencies: Record<string, number>;
}

export interface EvalRunRequest {
  dataset_path: string;
  output_path?: string;
}

export interface EvalDatasetSummary {
  name: string;
  path: string;
  records: number;
  sample_queries: string[];
  updated_at: number;
}

export interface EvalReportSummary {
  name: string;
  path: string;
  records: number;
  status: string;
  aggregate: Record<string, number>;
  updated_at: number;
}

export interface EvalResultItem {
  sample_id?: string;
  trace_id?: string;
  query?: string;
  answer_exact_match: number;
  reference_context_recall: number;
  retrieved_context_count: number;
  answer?: string;
  reference_answer?: string;
}

export interface EvalAggregate {
  answer_exact_match: number;
  reference_context_recall: number;
  retrieved_context_count_avg: number;
}

export interface EvalReportDetail {
  status: string;
  records: number;
  aggregate: EvalAggregate;
  results: EvalResultItem[];
}

export interface DiagnosisFinding {
  category: string;
  severity: string;
  rationale: string;
  suggested_actions: string[];
  evidence: Record<string, unknown>;
}

export interface DiagnosisResponse {
  target_type: string;
  trace_id?: string;
  sample_id?: string;
  summary: string;
  findings: DiagnosisFinding[];
  ai_suggestion?: string | null;
}

export interface EvalRunResponse {
  status: string;
  output_path?: string;
  report: EvalReportDetail;
}

export interface BenchmarkRunResponse {
  status: string;
  output_path?: string;
  report: Record<string, unknown>;
}

export interface BenchmarkReportSummary {
  name: string;
  path: string;
  records: number;
  status: string;
  aggregate: Record<string, number>;
  updated_at: number;
}

export interface WorkspaceSettings {
  kbId: string;
  tenantId: string;
  sessionId: string;
  apiKey: string;
}
