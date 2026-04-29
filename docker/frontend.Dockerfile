FROM node:24-alpine AS frontend-builder

ARG UI_REPO_URL=https://github.com/ifnodoraemon/nano-rag-ui.git
ARG UI_REF=main

RUN apk add --no-cache git

WORKDIR /frontend
RUN git clone --depth 1 --branch ${UI_REF} ${UI_REPO_URL} . \
 && (npm ci || npm install) \
 && npm run build

FROM nginx:1.27-alpine

COPY --from=frontend-builder /frontend/dist /usr/share/nginx/html
COPY docker/frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
