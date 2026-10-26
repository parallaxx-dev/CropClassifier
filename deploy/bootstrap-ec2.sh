#!/usr/bin/env bash
# One-time host setup for a fresh EC2 instance (Amazon Linux 2023 or
# Ubuntu 22.04/24.04): installs Docker Engine + the compose and buildx
# plugins, adds swap so the backend image build (torch + GDAL deps) doesn't
# OOM on small instances, and lets the login user run docker without sudo.
#
#   curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/<branch>/deploy/bootstrap-ec2.sh | sudo bash
#   # or, after cloning: sudo ./deploy/bootstrap-ec2.sh
#
# Safe to re-run. Log out and back in afterwards for the docker group to
# take effect.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run as root (sudo $0)" >&2
  exit 1
fi

SWAP_SIZE="${SWAP_SIZE:-2G}"
TARGET_USER="${SUDO_USER:-ec2-user}"

# shellcheck source=/dev/null
. /etc/os-release

case "$ID" in
  amzn)
    dnf install -y docker git
    # AL2023's docker package ships neither plugin -- install them from
    # Docker's release binaries.
    PLUGIN_DIR=/usr/local/lib/docker/cli-plugins
    mkdir -p "$PLUGIN_DIR"
    ARCH="$(uname -m)"
    if [[ ! -x "$PLUGIN_DIR/docker-compose" ]]; then
      curl -fsSL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-${ARCH}" \
        -o "$PLUGIN_DIR/docker-compose"
      chmod +x "$PLUGIN_DIR/docker-compose"
    fi
    if [[ ! -x "$PLUGIN_DIR/docker-buildx" ]]; then
      BUILDX_ARCH=$([[ "$ARCH" == "aarch64" ]] && echo arm64 || echo amd64)
      BUILDX_VERSION="$(curl -fsSL https://api.github.com/repos/docker/buildx/releases/latest | grep -m1 '"tag_name"' | cut -d'"' -f4)"
      curl -fsSL "https://github.com/docker/buildx/releases/download/${BUILDX_VERSION}/buildx-${BUILDX_VERSION}.linux-${BUILDX_ARCH}" \
        -o "$PLUGIN_DIR/docker-buildx"
      chmod +x "$PLUGIN_DIR/docker-buildx"
    fi
    ;;
  ubuntu|debian)
    apt-get update
    apt-get install -y ca-certificates curl git
    # Docker's official convenience script: Engine + compose + buildx plugins.
    if ! command -v docker >/dev/null; then
      curl -fsSL https://get.docker.com | sh
    fi
    ;;
  *)
    echo "Unsupported distro: $ID -- install Docker Engine + the compose plugin manually." >&2
    exit 1
    ;;
esac

systemctl enable --now docker
usermod -aG docker "$TARGET_USER"

if ! swapon --show | grep -q /swapfile; then
  fallocate -l "$SWAP_SIZE" /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

docker --version
docker compose version
echo
echo "Done. Log out and back in (docker group for $TARGET_USER), then follow DEPLOY.md."
