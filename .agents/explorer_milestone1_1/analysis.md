# Technical Design Analysis: Streaming Output & Frontend Integration (Milestone 2 - R1)

## 1. Executive Summary
This analysis details the technical design for introducing Server-Sent Events (SSE) streaming to the `nano-rag` platform under Milestone 2 (R1). The goal is to migrate the currently synchronous `/chat/stream` endpoint to a fully functional streaming endpoint, allowing the user interface to render response content progressively (typewriter effect) and display active reasoning steps in real-time.

A primary technical challenge is that the LLM generation in `AgenticReasoningService._answer_synthesis_node` generates structured JSON conforming to a strict JSON schema (containing `is_answerable`, `missing_entities`, `extracted_answer`, and `supporting_claims`). To render the answer progressively, the backend must parse this streaming JSON on-the-fly, extract the content of the `extracted_answer` field, and stream the clean tokens to the client via SSE, followed by the complete parsed object (including citations, claims, and trace metadata) once generation completes.

---

## 2. Component Architecture Overview

### 2.1 Backend Flow (Synchronous)
1. **API Router (`app/api/routes_business.py`)**: Receives POST request on `/v1/rag/chat`, extracts parameters, and calls `chat_pipeline.run(payload)`.
2. **LangGraph Agent Workflow (`app/agentic/service.py`)**: Runs `AgenticReasoningService` containing several nodes (`intent_decomposition` -> `initial_recall` -> `verification` -> `corrective_recall` -> `answer_synthesis`).
3. **Structured Generation (`app/agentic/service.py`)**: The `_answer_synthesis_node` requests a structured JSON schema response from `self.generation_client.generate`.
4. **Gateway Client (`app/model_client/base.py`)**: Issues a blocking POST request using `httpx.AsyncClient` to the LLM endpoint and returns the full JSON response.
5. **Formatting (`app/generation/answer_formatter.py`)**: Converts the plain text version of the structured JSON response into a `ChatResponse` with citations, supporting claims, and contexts.

### 2.2 Backend Flow (Proposed Streaming)
1. **API Router (`app/api/routes_business.py`)**: Receives POST request on `/v1/rag/chat/stream`, instantiates an `asyncio.Queue` and a streaming callback, and sets a context-local variable (`stream_callback_var`). Runs the workflow using LangGraph's `astream()` method to yield progressive status updates (e.g. searching, verifying).
2. **Gateway Client (`app/model_client/base.py` & `app/model_client/generation.py`)**: Launches a streaming POST request with `stream: true` using `httpx.AsyncClient.stream("POST", ...)` and yields raw tokens.
3. **On-the-Fly JSON Parsing (`app/agentic/service.py`)**: Inside `_answer_synthesis_node`, if `stream_callback_var` is set, the node processes LLM output using a stateful streaming JSON parser (`StreamingJsonFieldExtractor`) to extract the text of `extracted_answer` progressively, pushing chunks back to the API router queue.
4. **Final Response Yield**: When the stream completes, the full JSON is loaded. The standard formatting/tracing code runs unchanged, and the final metadata (citations, claims, trace ID) is sent as the last SSE event (`success`).

---

## 3. Streaming JSON Extraction Logic
Because the LLM response is returned as a JSON object, the raw stream contains chunks of JSON string syntax. The backend must extract the content of `"extracted_answer"` without waiting for the full response to finish.

We propose a stateful parser `StreamingJsonFieldExtractor`:
- **State**: `buffer` (accumulates raw stream), `started` (bool, whether `"extracted_answer": "` has been encountered), `finished` (bool, whether the closing quote of the field is reached), and `last_emitted_index`.
- **Search**: Scans the buffer using a regex pattern `r'"extracted_answer"\s*:\s*"'` to locate the start of the field.
- **Extraction**: Appends subsequent characters to `extracted_text`.
- **Unescaping & Safe Emission**: Detects the unescaped closing quote (ignoring `\"`). Emits new characters after unescaping common sequences like `\n` or `\t`, keeping at least 1 character in reserve to avoid splitting escape sequences across stream boundaries.

```python
import re

class StreamingJsonFieldExtractor:
    def __init__(self, target_key: str = "extracted_answer"):
        self.target_key = target_key
        self.buffer = ""
        self.started = False
        self.finished = False
        self.extracted_text = ""
        self.last_emitted_index = 0

    def feed(self, chunk: str) -> str:
        if self.finished:
            return ""
        self.buffer += chunk
        
        if not self.started:
            pattern = rf'"{re.escape(self.target_key)}"\s*:\s*"'
            match = re.search(pattern, self.buffer)
            if match:
                self.started = True
                start_index = match.end()
                self.extracted_text = self.buffer[start_index:]
                self.last_emitted_index = 0
            else:
                return ""
        else:
            self.extracted_text += chunk
            
        delta = ""
        i = self.last_emitted_index
        while i < len(self.extracted_text):
            char = self.extracted_text[i]
            if char == '"':
                # Check for escape sequences
                bs_count = 0
                idx = i - 1
                while idx >= 0 and self.extracted_text[idx] == '\\':
                    bs_count += 1
                    idx -= 1
                if bs_count % 2 == 0:
                    delta += self.extracted_text[self.last_emitted_index:i]
                    self.finished = True
                    break
            i += 1
            
        if not self.finished:
            safe_len = len(self.extracted_text)
            if safe_len > 0 and self.extracted_text[-1] == '\\':
                safe_len -= 1
            if safe_len > self.last_emitted_index:
                emit_end = safe_len
                if emit_end > self.last_emitted_index + 1:
                    emit_end -= 1  # Leave lookahead room
                chunk_to_emit = self.extracted_text[self.last_emitted_index:emit_end]
                chunk_to_emit = chunk_to_emit.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')
                delta += chunk_to_emit
                self.last_emitted_index = emit_end
                
        return delta
```

---

## 4. Proposed Source Code Changes

### 4.1 Backend Changes

#### File 1: `app/model_client/base.py`
Add `chat_completions_stream` to `AsyncJsonProviderClient`:
```python
    async def chat_completions_stream(
        self, messages: list[dict[str, Any]], model_alias: str | None = None, **extra: Any
    ) -> AsyncGenerator[str, None]:
        self.provider.require_ready(require_model=False)
        payload = {
            "model": model_alias or self.provider.model,
            "messages": messages,
            "stream": True,
            **extra,
        }
        client = self._get_client()
        url = f"{self.provider.base_url}/chat/completions"
        async with client.stream("POST", url, headers=self.headers, json=payload) as response:
            if response.status_code >= 400:
                detail = await response.aread()
                raise ModelGatewayError(
                    f"{self.provider.capability} provider stream failed: "
                    f"{response.status_code} {detail.decode('utf-8', errors='ignore')}"
                )
            async for line in response.aiter_lines():
                line = line.strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        logger.warning("Failed to decode stream line: %s", line)
```

#### File 2: `app/model_client/generation.py`
Expose the stream method in `GenerationClient`:
```python
    async def generate_stream(
        self, messages: list[dict[str, Any]], model_alias: str | None = None, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        async for chunk in self.provider_client.chat_completions_stream(
            messages, model_alias or self.alias, **kwargs
        ):
            yield chunk
```

#### File 3: `app/agentic/service.py`
- Define a context-local variable: `stream_callback_var = contextvars.ContextVar("stream_callback", default=None)`.
- Embed the `StreamingJsonFieldExtractor` class.
- Modify `_answer_synthesis_node` to handle streaming:
```python
        callback = stream_callback_var.get()
        if callback:
            full_content_list = []
            extractor = StreamingJsonFieldExtractor("extracted_answer")
            async for chunk in self.generation_client.generate_stream(messages, response_format=schema):
                full_content_list.append(chunk)
                delta = extractor.feed(chunk)
                if delta:
                    await callback(delta)
            raw_content = "".join(full_content_list).strip()
            # Prepare result dictionary structure expected by existing parsing logic
            result = {
                "content": raw_content,
                "finish_reason": "stop",
                "usage": {},
                "model": self.generation_client.alias,
                "raw": {}
            }
        else:
            result = await self.generation_client.generate(messages, response_format=schema)
```

#### File 4: `app/api/routes_business.py`
Refactor the `/chat/stream` endpoint implementation:
```python
import contextvars
from app.agentic.service import stream_callback_var

@router.post("/chat/stream")
async def rag_chat_stream(
    payload: BusinessChatRequest,
    request: Request,
    context: RequestContext = Depends(require_api_key),
) -> StreamingResponse:
    container = request.app.state.container
    _ensure_kb_access(container, payload.kb_id, context)
    
    async def event_generator():
        token_queue = asyncio.Queue()
        
        async def on_token(token: str):
            await token_queue.put(token)
            
        token_reset_key = stream_callback_var.set(on_token)
        chat_req = ChatRequest(
            query=payload.query,
            top_k=payload.top_k,
            kb_id=payload.kb_id,
            session_id=payload.session_id,
            metadata_filters=payload.metadata_filters,
        )
        
        # Mapping LangGraph nodes to user-facing messages
        status_messages = {
            "intent_decomposition": "Decomposing search intent...",
            "initial_recall": "Searching database for initial contexts...",
            "verification": "Verifying content sufficiency...",
            "corrective_recall": "Optimizing retrieval and expanding search...",
            "answer_synthesis": "Generating response..."
        }
        
        response_container = {}
        
        async def run_pipeline():
            try:
                # Use astream to receive updates as each node executes
                async for event in container.chat_pipeline.workflow.astream(
                    {"payload": chat_req, "started_at": perf_counter()}
                ):
                    node_name = next(iter(event.keys()))
                    await token_queue.put({"type": "status", "node": node_name})
                    if "response" in event[node_name]:
                        response_container["response"] = event[node_name]["response"]
            except Exception as e:
                logger.exception("Error during pipeline execution")
                await token_queue.put({"type": "error", "message": str(e)})
            finally:
                await token_queue.put(None)
                stream_callback_var.reset(token_reset_key)
                
        pipeline_task = asyncio.create_task(run_pipeline())
        
        while True:
            item = await token_queue.get()
            if item is None:
                break
            if isinstance(item, str):
                yield f"data: {json.dumps({'status': 'generating', 'delta': item})}\n\n"
            elif isinstance(item, dict):
                if item["type"] == "status":
                    node = item["node"]
                    msg = status_messages.get(node, f"Processing node {node}...")
                    yield f"data: {json.dumps({'status': 'thinking', 'node': node, 'message': msg})}\n\n"
                elif item["type"] == "error":
                    yield f"data: {json.dumps({'status': 'error', 'message': item['message']})}\n\n"
            token_queue.task_done()
            
        await pipeline_task
        
        if "response" in response_container:
            res = response_container["response"]
            yield f"data: {json.dumps({
                'status': 'success',
                'answer': res.answer,
                'citations': [c.model_dump() for c in res.citations],
                'contexts': res.contexts,
                'trace_id': res.trace_id,
                'kb_id': payload.kb_id,
                'session_id': payload.session_id
            })}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

---

### 4.2 Frontend Changes

#### File 5: `frontend/src/lib/api.ts`
Introduce the streaming event contract and client call logic using the browser `ReadableStream` reader interface:
```typescript
export interface StreamEvent {
  status: 'thinking' | 'generating' | 'success' | 'error';
  node?: string;
  message?: string;
  delta?: string;
  answer?: string;
  citations?: Citation[];
  contexts?: any[];
  trace_id?: string;
}

export async function chatStream(
  payload: ChatRequest,
  onEvent: (event: StreamEvent) => void
): Promise<void> {
  eventBus.emit(`正在发送流式问答请求`, 'info');
  
  const response = await fetch('/v1/rag/chat/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || `HTTP ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('Response body is not readable');
  }

  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Hold onto incomplete lines

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        if (trimmed.startsWith('data: ')) {
          try {
            const event: StreamEvent = JSON.parse(trimmed.slice(6));
            onEvent(event);
          } catch (e) {
            console.error('Failed to parse SSE event:', trimmed, e);
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
```

#### File 6: `frontend/src/components/ChatInterface.tsx`
- Extend the React component's `Message` interface:
```typescript
interface Message {
  id: string;
  role: 'user' | 'model';
  text: string;
  trace?: RetrievalTrace;
  feedback?: 'up' | 'down' | null;
  status?: 'thinking' | 'generating' | 'success' | 'error';
  statusText?: string;
}
```
- Refactor `handleSubmit` to invoke `chatStream`, updating state progressively:
```typescript
  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    const query = input.trim();
    if (!query || isLoading || !settings.kbId) return;

    setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: 'user', text: query }]);
    setInput('');
    setIsLoading(true);
    const startTime = performance.now();
    const modelMessageId = crypto.randomUUID();

    setMessages((prev) => [
      ...prev,
      { id: modelMessageId, role: 'model', text: '', status: 'thinking', statusText: 'Initializing...' }
    ]);

    let accumulatedText = '';

    try {
      await chatStream({
        query,
        kb_id: settings.kbId,
        session_id: settings.sessionId,
        top_k: settings.topK,
      }, (event) => {
        if (event.status === 'thinking') {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === modelMessageId
                ? { ...msg, statusText: event.message || 'Thinking...' }
                : msg
            )
          );
        } else if (event.status === 'generating') {
          if (event.delta) {
            accumulatedText += event.delta;
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === modelMessageId
                  ? { ...msg, text: accumulatedText, status: 'generating', statusText: 'Generating answer...' }
                  : msg
              )
            );
          }
        } else if (event.status === 'success') {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === modelMessageId
                ? {
                    ...msg,
                    text: event.answer || accumulatedText,
                    status: 'success',
                    statusText: '',
                    trace: {
                      latency: Math.round(performance.now() - startTime),
                      trace_id: event.trace_id,
                      results: event.citations || [],
                    },
                    feedback: null,
                  }
                : msg
            )
          );
          setIsLoading(false);
        } else if (event.status === 'error') {
          throw new Error(event.message || 'Error occurred during streaming.');
        }
      });
    } catch (error) {
      console.error(error);
      const detail = error instanceof Error ? error.message : String(error);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === modelMessageId
            ? {
                ...msg,
                text: `请求失败：${detail}`,
                status: 'error',
                statusText: '',
              }
            : msg
        )
      );
      setIsLoading(false);
    }
  };
```
- Update loading state feedback inside the JSX render tree:
```typescript
        {isLoading && (
          <div className="mb-5 flex gap-3">
            <div className="mt-1 flex h-9 w-9 items-center justify-center rounded-lg border border-emerald-200 bg-emerald-50 text-emerald-700">
              <Loader2 className="h-4 w-4 animate-spin" />
            </div>
            <div className="rounded-lg border border-slate-200 bg-white px-4 py-3 text-sm text-slate-500 shadow-sm">
              {messages.find(m => m.id === messages[messages.length - 1]?.id)?.statusText || '正在检索证据并生成回答...'}
            </div>
          </div>
        )}
```

---

## 5. Verification Plan

### 5.1 Backend Verification
1. Run local dev server: `uvicorn app.main:app --reload`.
2. Issue a manual POST request using `curl` to the stream endpoint:
   ```bash
   curl -N -X POST http://localhost:8000/v1/rag/chat/stream \
     -H "Content-Type: application/json" \
     -H "X-API-Key: test-key" \
     -d '{"query": "什么是系统管理？", "kb_id": "default"}'
   ```
3. Verify that the output consists of consecutive lines starting with `data: ` and containing valid JSON structure:
   - Starts with `status: "thinking"` blocks showing transition of workflow nodes.
   - Shows `status: "generating"` containing incremental `delta` values of the answer text.
   - Concludes with `status: "success"` containing the full payload (trace_id, answer, citations, etc.).

### 5.2 Frontend Verification
1. Open the dev environment, ask a question in the RAG chat input.
2. Verify that the loading indicator changes content progressively (e.g. from "Decomposing search intent..." to "Searching database...").
3. Verify that the answer text begins to stream word-by-word into the message bubble.
4. Verify that once the stream finishes, the "证据" (Citations) button and metrics (latency, trace ID) appear underneath the message.
