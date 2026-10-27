#!/usr/bin/env bash
# Instance-side half of deploy/deploy-ec2.sh -- shipped inside each release
# and run over SSH. Not meant to be run by hand, except `status`:
#
#   remote.sh activate <sha> <port> <keep> deploy|rollback
#   remote.sh status
set -euo pipefail

BASE="${CROP_BASE:-$HOME/cropclassifier}"
PROJECT=cropclassifier
COMPOSE_VERSION=v2.32.4
BUILDX_VERSION=v0.20.1

log() { printf '\033[1;32m[remote]\033[0m %s\n' "$*"; }

ensure_docker() {
  if ! command -v docker >/dev/null; then
    . /etc/os-release
    log "Installing Docker ($ID)"
    case "$ID" in
      amzn)
        sudo dnf install -y docker
        ;;
      ubuntu|debian)
        curl -fsSL https://get.docker.com | sudo sh
        ;;
      *)
        echo "Unsupported OS '$ID' -- install Docker + the compose plugin manually, then re-run." >&2
        exit 1
        ;;
    esac
    sudo systemctl enable --now docker
    sudo usermod -aG docker "$USER"
  fi

  # Group membership from usermod only applies to new logins, so fall back
  # to sudo for this session.
  if docker info >/dev/null 2>&1; then DOCKER=(docker); else DOCKER=(sudo docker); fi

  # Amazon Linux's docker package ships neither the compose nor buildx plugin.
  local arch plugins=/usr/local/lib/docker/cli-plugins
  arch="$(uname -m)"
  if ! "${DOCKER[@]}" compose version >/dev/null 2>&1; then
    log "Installing docker compose $COMPOSE_VERSION"
    sudo mkdir -p "$plugins"
    sudo curl -fsSL -o "$plugins/docker-compose" \
      "https://github.com/docker/compose/releases/download/$COMPOSE_VERSION/docker-compose-linux-$arch"
    sudo chmod +x "$plugins/docker-compose"
  fi
  if ! "${DOCKER[@]}" buildx version >/dev/null 2>&1; then
    local barch=amd64
    [[ "$arch" == aarch64 ]] && barch=arm64
    log "Installing docker buildx $BUILDX_VERSION"
    sudo mkdir -p "$plugins"
    sudo curl -fsSL -o "$plugins/docker-buildx" \
      "https://github.com/docker/buildx/releases/download/$BUILDX_VERSION/buildx-$BUILDX_VERSION.linux-$barch"
    sudo chmod +x "$plugins/docker-buildx"
  fi
}

ensure_swap() {
  local mem_kb
  mem_kb="$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
  if (( mem_kb < 3000000 )) && [[ -z "$(swapon --noheadings 2>/dev/null)" ]]; then
    log "Low memory (${mem_kb}kB), adding a 2GB swapfile for the image build"
    sudo fallocate -l 2G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile >/dev/null
    sudo swapon /swapfile
    grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
  fi
}

# Points the release's gitignored paths at the shared data that persists
# across releases (compose bind-mounts / env_file follow the symlinks).
link_shared_data() {
  local rel="$1"
  mkdir -p "$rel/backend/training/eurocrops_output/multi_country"
  ln -sfn "$BASE/shared/checkpoints" "$rel/backend/training/eurocrops_output/multi_country/checkpoints"
  ln -sfn "$BASE/shared/local_parcels" "$rel/backend/local_parcels"
  if [[ ! -f "$BASE/shared/backend.env" ]]; then
    echo "error: $BASE/shared/backend.env missing -- deploy once without --skip-data" >&2
    exit 1
  fi
  ln -sfn "$BASE/shared/backend.env" "$rel/backend/.env"
}

compose_up() {
  local rel="$1" sha="$2" port="$3" build="$4"
  # compose reads this .env for variable substitution in docker-compose.yml
  printf 'APP_VERSION=%s\nFRONTEND_PORT=%s\nBACKEND_PORT=%s\n' \
    "$sha" "$port" "${BACKEND_PORT:-9000}" > "$rel/.env"
  (cd "$rel" && "${DOCKER[@]}" compose -p "$PROJECT" up -d --remove-orphans $build)
}

wait_healthy() {
  local port="$1"
  for _ in $(seq 1 60); do
    if curl -fsS -o /dev/null "http://localhost:$port/api/v1/model-info" 2>/dev/null; then
      return 0
    fi
    sleep 5
  done
  return 1
}

prune_releases() {
  local keep="$1" current_sha="$2" old
  # Newest first by mtime; never remove the active release.
  for old in $(ls -1t "$BASE/releases" | tail -n +"$((keep + 1))"); do
    [[ "$old" == "$current_sha" ]] && continue
    log "Removing old release $old"
    rm -rf "${BASE:?}/releases/$old"
    "${DOCKER[@]}" rmi "cropclassifier-backend:$old" "cropclassifier-frontend:$old" >/dev/null 2>&1 || true
  done
  "${DOCKER[@]}" image prune -f >/dev/null
}

activate() {
  local sha="$1" port="$2" keep="$3" mode="$4"
  local rel="$BASE/releases/$sha"
  [[ -d "$rel" ]] || { echo "error: release $sha not found in $BASE/releases" >&2; exit 1; }

  local previous=""
  [[ -L "$BASE/current" ]] && previous="$(basename "$(readlink "$BASE/current")")"

  ensure_docker
  ensure_swap
  link_shared_data "$rel"

  # A rollback reuses the image tagged for that release if it still exists
  # (compose builds it only when missing); a deploy always rebuilds.
  local build="--build"
  [[ "$mode" == rollback ]] && build=""

  log "Starting release $sha"
  if compose_up "$rel" "$sha" "$port" "$build" && wait_healthy "$port"; then
    ln -sfn "$rel" "$BASE/current"
    touch "$rel"
    echo "$(date -u +%FT%TZ) $mode $sha" >> "$BASE/deploy-history.log"
    log "Release $sha is live and healthy"
    prune_releases "$keep" "$sha"
    return 0
  fi

  echo "error: release $sha failed to start or never became healthy" >&2
  (cd "$rel" && "${DOCKER[@]}" compose -p "$PROJECT" logs --tail 40) || true
  echo "$(date -u +%FT%TZ) failed-$mode $sha" >> "$BASE/deploy-history.log"
  if [[ -n "$previous" && "$previous" != "$sha" && -d "$BASE/releases/$previous" ]]; then
    log "Restoring previous release $previous"
    compose_up "$BASE/releases/$previous" "$previous" "$port" "" || true
    echo "$(date -u +%FT%TZ) auto-rollback $previous" >> "$BASE/deploy-history.log"
  fi
  exit 1
}

status() {
  if docker info >/dev/null 2>&1; then DOCKER=(docker); else DOCKER=(sudo docker); fi
  if [[ -L "$BASE/current" ]]; then
    echo "Current release: $(basename "$(readlink "$BASE/current")")"
  else
    echo "No release deployed yet."
  fi
  echo "Releases on disk: $(ls -1t "$BASE/releases" 2>/dev/null | tr '\n' ' ')"
  echo "Recent history:"
  tail -n 10 "$BASE/deploy-history.log" 2>/dev/null | sed 's/^/  /' || true
  [[ -L "$BASE/current" ]] && (cd "$BASE/current" && "${DOCKER[@]}" compose -p "$PROJECT" ps)
}

case "${1:-}" in
  activate) shift; activate "$@" ;;
  status) status ;;
  *) echo "usage: $0 activate <sha> <port> <keep> deploy|rollback | status" >&2; exit 2 ;;
esac
