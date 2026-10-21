FROM node:24-alpine AS frontend-builder

WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend ./
RUN npm run build

FROM nginx:1.27-alpine

COPY --from=frontend-builder /frontend/dist /usr/share/nginx/html
COPY docker/frontend/nginx.conf /etc/nginx/templates/default.conf.template
# Shared per-location proxy headers: strips client-supplied proxy identity
# headers on every proxied route (the public port is not a trusted proxy).
COPY docker/frontend/proxy-common.conf /etc/nginx/snippets/proxy-common.conf

EXPOSE 80
