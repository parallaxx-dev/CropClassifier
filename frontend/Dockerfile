# Multi-stage: build the static bundle with Node, serve it with nginx --
# the final image ships no Node runtime, just the built assets + a tiny
# static file server.

FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .

# Vite inlines import.meta.env.VITE_* at BUILD time, not read at container
# runtime -- must be passed as a build arg, not a docker-run env var (see
# src/api.js). Defaults to the relative /api/v1 path, which nginx below
# proxies to the backend, so the same bundle works on any host/domain.
ARG VITE_API_BASE=/api/v1
ENV VITE_API_BASE=$VITE_API_BASE
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
# Rendered to /etc/nginx/conf.d/default.conf at container start, with
# BACKEND_UPSTREAM substituted (see nginx/default.conf.template).
COPY nginx/default.conf.template /etc/nginx/templates/default.conf.template
ENV BACKEND_UPSTREAM=backend:8000
EXPOSE 80
