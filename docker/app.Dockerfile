FROM python:3.12-slim@sha256:7026274c107626d7e940e0e5d6730481a4600ae95d5ca7eb532dd4180313fea9

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY configs ./configs
COPY scripts ./scripts
COPY data/raw ./data/raw
COPY data/eval ./data/eval
COPY data/samples ./data/samples
COPY frontend/dist ./frontend/dist

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
