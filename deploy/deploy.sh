#!/usr/bin/env bash
# (Re)deploy the app on this host: pull the latest code, rebuild images,
# restart containers, and wait until the API answers through nginx.
# Idempotent -- used both by hand and by the CD workflow over SSH.
#
#   ./deploy/deploy.sh            # pull current branch, rebuild, restart
#   ./deploy/deploy.sh --no-pull  # rebuild/restart the checked-out code as-is
#
# Picks up the HTTPS overlay automatically when DOMAIN is set in .env.
set -euo pipefail

cd "$(dirname "$0")/.."

PULL=1
[[ "${1:-}" == "--no-pull" ]] && PULL=0

if [[ ! -f backend/.env ]]; then
  echo "backend/.env is missing -- copy backend/.env.example and fill in the CDSE credentials." >&2
  exit 1
fi

# Read only the keys this script needs from .env rather than sourcing it.
env_value() {
  [[ -f .env ]] || return 0
  # `|| true`: an absent key is normal, not an error under pipefail.
  { grep -E "^$1=" .env || true; } | tail -1 | cut -d= -f2- | tr -d '"'"'"
}
DOMAIN="$(env_value DOMAIN)"
HTTP_PORT="$(env_value HTTP_PORT)"

COMPOSE=(docker compose -f docker-compose.yml)
if [[ -n "$DOMAIN" ]]; then
  COMPOSE+=(-f docker-compose.https.yml)
  HEALTH_URL="https://${DOMAIN}/health"
else
  HEALTH_URL="http://localhost:${HTTP_PORT:-80}/health"
fi

if [[ $PULL -eq 1 ]]; then
  git pull --ff-only
fi

"${COMPOSE[@]}" up -d --build --remove-orphans

# Rebuilds leave the previous images dangling -- reclaim the disk.
docker image prune -f >/dev/null

echo "Waiting for $HEALTH_URL ..."
for _ in $(seq 1 30); do
  if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
    echo "Deployed: $HEALTH_URL is healthy."
    "${COMPOSE[@]}" ps
    exit 0
  fi
  sleep 5
done

echo "Health check failed after 150s. Recent logs:" >&2
"${COMPOSE[@]}" logs --tail 50 >&2
exit 1
