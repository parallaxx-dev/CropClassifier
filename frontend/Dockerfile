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
# src/api.js). Defaults to the local dev backend port if not overridden.
ARG VITE_API_BASE=http://localhost:9000/api/v1
ENV VITE_API_BASE=$VITE_API_BASE
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
