#!/usr/bin/env bash
# Deploys a specific git commit of this repo to an EC2 instance (or any
# Linux host reachable over SSH) and runs it with docker compose. Run from
# your own machine, inside the repo:
#
#   deploy/deploy-ec2.sh --host ec2-user@1.2.3.4 --key ~/.ssh/my-key.pem
#   deploy/deploy-ec2.sh --host ubuntu@1.2.3.4 --ref v1.2.0
#   deploy/deploy-ec2.sh --host ubuntu@1.2.3.4 --rollback 78ace17abcd1
#   deploy/deploy-ec2.sh --host ubuntu@1.2.3.4 --status
#
# Versioning: every deploy is an exact git commit, shipped via `git archive`
# (so uncommitted changes are never deployed and the instance needs no
# GitHub access). On the instance, under ~/cropclassifier/:
#   releases/<sha>/     one directory per deployed commit
#   current -> releases/<sha>
#   shared/             data that outlives releases: backend.env (CDSE
#                       credentials), parcel geometry, local_parcels
#   deploy-history.log  what was deployed/rolled back, and when
# Images are tagged cropclassifier-{backend,frontend}:<sha>. If the new
# release fails its health check, the previous release is restored.
#
# The gitignored runtime data (backend/.env, parcel *.geojson, local_parcels/)
# is rsynced from this machine on each deploy unless --skip-data is passed.
#
# Instance prerequisites: SSH access with passwordless sudo (default for
# ec2-user / ubuntu), outbound internet, and the security group allowing
# inbound TCP on --port (default 80). Docker + compose are installed
# automatically on first deploy (Amazon Linux 2023 / Ubuntu / Debian).
# Use >= 2GB RAM (t3.small+); a swapfile is added on smaller instances so
# the PyTorch install during the backend build doesn't run out of memory.
set -euo pipefail

usage() {
  sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'
  cat <<'EOF'
Options:
  -H, --host USER@HOST   SSH target (or set DEPLOY_HOST)
  -i, --key PATH         SSH private key (or set DEPLOY_KEY)
  -r, --ref REF          git commit/tag/branch to deploy (default: HEAD)
  -p, --port PORT        public port for the web UI (default: 80)
      --keep N           number of releases to keep on the instance (default: 5)
      --skip-data        don't re-sync .env / parcel data
      --rollback SHA     switch back to an already-deployed release
      --status           show the current release and container status
  -h, --help             show this help
EOF
}

HOST="${DEPLOY_HOST:-}"
KEY="${DEPLOY_KEY:-}"
REF="HEAD"
PORT=80
KEEP=5
SKIP_DATA=0
ACTION=deploy
ROLLBACK_SHA=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -H|--host) HOST="$2"; shift 2 ;;
    -i|--key) KEY="$2"; shift 2 ;;
    -r|--ref) REF="$2"; shift 2 ;;
    -p|--port) PORT="$2"; shift 2 ;;
    --keep) KEEP="$2"; shift 2 ;;
    --skip-data) SKIP_DATA=1; shift ;;
    --rollback) ACTION=rollback; ROLLBACK_SHA="$2"; shift 2 ;;
    --status) ACTION=status; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$HOST" ]] || { echo "error: --host is required (e.g. ec2-user@1.2.3.4)" >&2; exit 2; }
for tool in git ssh rsync; do
  command -v "$tool" >/dev/null || { echo "error: '$tool' not found on this machine" >&2; exit 1; }
done

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Paths on the instance are relative to the SSH user's home directory.
BASE=cropclassifier

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
[[ -n "$KEY" ]] && SSH_OPTS+=(-i "$KEY")
remote() { ssh "${SSH_OPTS[@]}" "$HOST" "$@"; }
RSYNC_SSH="ssh ${SSH_OPTS[*]}"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }

if [[ "$ACTION" == status ]]; then
  remote "bash $BASE/current/deploy/remote.sh status"
  exit 0
fi

if [[ "$ACTION" == rollback ]]; then
  # Release directories are named by 12-char SHA; normalize whatever was given.
  ROLLBACK_SHA="$(git rev-parse --short=12 "$ROLLBACK_SHA^{commit}" 2>/dev/null || echo "$ROLLBACK_SHA")"
  log "Rolling back $HOST to $ROLLBACK_SHA"
  remote "bash $BASE/releases/$ROLLBACK_SHA/deploy/remote.sh activate $ROLLBACK_SHA $PORT $KEEP rollback"
  exit 0
fi

# ---- deploy ----
SHA="$(git rev-parse --short=12 "$REF^{commit}")"
if [[ "$REF" == HEAD ]] && [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "warning: uncommitted changes are NOT deployed -- deploying commit $SHA as committed." >&2
fi
if [[ "$SKIP_DATA" == 0 ]] && [[ ! -f backend/.env ]]; then
  echo "error: backend/.env not found (copy backend/.env.example and fill it in, or pass --skip-data)" >&2
  exit 1
fi

log "Deploying $SHA ($(git log -1 --format=%s "$SHA")) to $HOST"

log "Uploading release"
remote "mkdir -p $BASE/releases/$SHA $BASE/shared/checkpoints $BASE/shared/local_parcels"
git archive --format=tar "$SHA" | gzip | remote "tar -xzf - -C $BASE/releases/$SHA"

if [[ "$SKIP_DATA" == 0 ]]; then
  log "Syncing credentials and parcel data"
  rsync -az --chmod=F600 -e "$RSYNC_SSH" backend/.env "$HOST:$BASE/shared/backend.env"
  CHECKPOINTS=backend/training/eurocrops_output/multi_country/checkpoints
  if [[ -d "$CHECKPOINTS" ]]; then
    rsync -az --delete --include='*_sampled_geoms.geojson' --exclude='*' -e "$RSYNC_SSH" \
      "$CHECKPOINTS/" "$HOST:$BASE/shared/checkpoints/"
  else
    echo "warning: $CHECKPOINTS missing locally -- Parcel Browser will have no EuroCrops/India parcels" >&2
  fi
  if [[ -d backend/local_parcels ]]; then
    rsync -az --delete -e "$RSYNC_SSH" backend/local_parcels/ "$HOST:$BASE/shared/local_parcels/"
  fi
fi

log "Building and starting on the instance (first deploy installs Docker and takes a while)"
remote "bash $BASE/releases/$SHA/deploy/remote.sh activate $SHA $PORT $KEEP deploy"

PUBLIC_HOST="${HOST#*@}"
log "Done: http://$PUBLIC_HOST$([[ "$PORT" == 80 ]] || echo ":$PORT")/  (release $SHA)"
