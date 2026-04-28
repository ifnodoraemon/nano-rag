FROM node:24-alpine AS frontend-builder

WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend ./
RUN npm run build

FROM python:3.12-slim@sha256:7026274c107626d7e940e0e5d6730481a4600ae95d5ca7eb532dd4180313fea9

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

COPY app ./app
COPY configs ./configs
COPY scripts ./scripts
COPY --from=frontend-builder /frontend/dist ./frontend/dist

RUN mkdir -p \
    /workspace/data/parsed \
    /workspace/data/reports/traces \
    /workspace/data/reports/feedback \
    /workspace/data/uploads
RUN chown -R appuser:appgroup /workspace

USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
