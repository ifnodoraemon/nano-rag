# Handoff Report: Streaming Output & Frontend Integration (Milestone 2 - R1)

## 1. Observation
We observed the following code components in the `nano-rag` codebase:
- **Mock `/chat/stream` Endpoint**: Located in `app/api/routes_business.py` (lines 289–316), the endpoint currently yields a mock thinking message, runs the entire synchronous pipeline, and dumps the final answer:
  ```python
  @router.post("/chat/stream")
  async def rag_chat_stream(
      payload: BusinessChatRequest,
      request: Request,
      context: RequestContext = Depends(require_api_key),
  ) -> StreamingResponse:
      ...
      async def event_generator():
          yield "data: {\"status\": \"thinking\", \"message\": \"Retrieving and synthesizing...\"}\n\n"
          # Temporarily call the synchronous pipeline and dump at the end
          response = await container.chat_pipeline.run(...)
          yield f"data: {{\"status\": \"success\", \"answer\": {json.dumps(response.answer)}, \"trace_id\": \"{response.trace_id}\"}}\n\n"
      return StreamingResponse(event_generator(), media_type="text/event-stream")
  ```
- **Synchronous Generation Client**: In `app/model_client/base.py` (lines 141–149), the `AsyncJsonProviderClient.chat_completions` executes `await self.post_json(...)` which returns a fully loaded JSON object. No streaming method exists.
- **Strict JSON Generation Schema**: In `app/agentic/service.py` (lines 187–218), `_answer_synthesis_node` generates structured JSON content conforming to a strict JSON schema:
  ```python
  schema = {
      "type": "json_schema",
      "json_schema": {
          "name": "answer_structure",
          "schema": {
              "type": "object",
              "properties": {
                  "is_answerable": { ... },
                  "missing_entities": { ... },
                  "extracted_answer": {
                      "type": "string",
                      "description": "带引用的极简答案，例如：xxx为xxx [C1]。"
                  },
                  "supporting_claims": { ... }
              },
              "required": ["is_answerable", "missing_entities", "extracted_answer", "supporting_claims"],
              "additionalProperties": False
          },
          "strict": True
      }
  }
  result = await self.generation_client.generate(messages, response_format=schema)
  ```
- **Static Frontend Chat Call**: In `frontend/src/lib/api.ts` (lines 131–140), `chat` executes a standard `fetch` call and parses it via `.json()`.
- **Static Chat UI Handling**: In `frontend/src/components/ChatInterface.tsx` (lines 42–87), `handleSubmit` calls `chat(...)` and blocks until the full response is available, rendering it in a single update.

---

## 2. Logic Chain
1. To stream the response progressively (typewriter effect), `/chat/stream` must be fully implemented to return standard Server-Sent Events (`text/event-stream`).
2. The LLM outputs a structured JSON object rather than raw plaintext. Therefore, streaming the raw LLM output directly to the client is insufficient because the client would receive chunks of raw JSON syntax (e.g. `{"extracted_answer": "Hello...`).
3. To resolve this, the backend must use an on-the-fly streaming parser to extract the `extracted_answer` string value as it streams from the LLM, and emit these text tokens progressively.
4. To allow the agent pipeline to notify the FastAPI router of streaming tokens without mutating LangGraph's state structures or schema parameters, we should use a thread/coroutine-local context variable (`contextvars.ContextVar`).
5. As LangGraph runs, we can utilize `workflow.astream()` to capture node transitions and emit corresponding `thinking` messages to the frontend.
6. When the LLM stream completes, we have collected all raw chunks to reconstruct the full JSON. This allows existing trace, citation, and claim formatting/logging code in the `_answer_synthesis_node` to execute unchanged, returning the full metadata packet in a final `success` event.
7. On the frontend, `chatStream` will be introduced to handle the SSE stream using a `ReadableStream` reader. `ChatInterface.tsx` will be modified to handle progressive updates and dynamically render step descriptions (e.g., "Decomposing search intent...") and typewriter text.

---

## 3. Caveats
- **JSON Schema Syntax Formatting**: The streaming JSON parser assumes that the LLM provider respects the response format constraint and generates valid JSON. If the LLM generates syntax errors or invalid JSON, the parser will fail to find `extracted_answer` or fail on closing quotes. To mitigate this, a final fallback event containing the full formatted answer (parsed from the complete string) must always be emitted at the end of the stream.
- **Provider Stream Capability**: We assume the upstream LLM gateway provider (e.g., OpenAI compatible) supports streaming for structured JSON output (`json_schema`) and forwards content chunks in `choices[0].delta.content`.

---

## 4. Conclusion
We have prepared a complete, self-contained technical design for Milestone 2 (R1): Streaming Output & Frontend Integration.
The following target files will be modified:
1. `app/model_client/base.py` (Add streaming capability to the HTTP gateway client).
2. `app/model_client/generation.py` (Expose stream method).
3. `app/agentic/service.py` (Add context variables, streaming JSON parser, and modify synthesis node).
4. `app/api/routes_business.py` (Replace stub `/chat/stream` with SSE generator).
5. `frontend/src/lib/api.ts` (Add SSE reader stream API client).
6. `frontend/src/components/ChatInterface.tsx` (Add typewriter state updates and agent progress).

---

## 5. Verification Method
- **Test Commands**:
  - Run the python service: `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
  - Execute a stream test via curl:
    ```bash
    curl -N -X POST http://localhost:8000/v1/rag/chat/stream \
      -H "Content-Type: application/json" \
      -H "X-API-Key: test-key" \
      -d '{"query": "什么是系统管理？", "kb_id": "default"}'
    ```
  - Verify that the SSE response starts with `data: {"status": "thinking", ...}` updates, then transitions to `data: {"status": "generating", "delta": ...}` tokens, and finishes with a `data: {"status": "success", ...}` message.
- **Frontend Verification**:
  - Ask a question in the UI.
  - Verify that the loader shows active progress descriptions (e.g., "Decomposing search intent...", "Searching database...").
  - Confirm the typewriter effect renders the text progressively.
  - Confirm citations and feedback buttons appear at the end of the text.
- **Invalidation Conditions**: If any of the SSE events do not start with `data: `, fail to parse as JSON, or if the typewriter text is empty or missing citations, the verification fails.
