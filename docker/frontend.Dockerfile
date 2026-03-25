FROM python:3.12-slim

WORKDIR /app

COPY frontend/dist /app/dist

EXPOSE 80

CMD ["python", "-m", "http.server", "80", "--directory", "/app/dist"]
