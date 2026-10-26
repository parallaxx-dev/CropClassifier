# Deploying CropClassifier

The whole app is two containers run by Docker Compose. The same setup
works on a laptop or a single EC2 instance:

```
browser ──► frontend (nginx, :HTTP_PORT) ──/api/, /health──► backend (FastAPI :8000, internal only)
                                                               ├─ model checkpoint (baked into image)
                                                               ├─ parcel data (bind-mounted, read-only)
                                                               └─ Copernicus CDSE (outbound HTTPS)
```

- `docker-compose.yml` is the base stack, serving plain HTTP on `HTTP_PORT`.
- `docker-compose.https.yml` is an optional overlay that puts Caddy in front for automatic Let's Encrypt HTTPS.
- `deploy/deploy.sh` does the pull, rebuild, restart and health check. It's the only deploy path; CD runs the same script.
- `deploy/bootstrap-ec2.sh` is the one-time Docker install on a fresh instance.

## What has to be on the host

| What | Where | In git? |
|---|---|---|
| Code | repo checkout | yes |
| CDSE Sentinel Hub credentials | `backend/.env` (from `backend/.env.example`) | **no** (secret) |
| Parcel geometries for Parcel Browser / Live Validation | `backend/training/eurocrops_output/multi_country/checkpoints/*_sampled_geoms.geojson` (~17 MB) | **no** |
| Hand-labelled custom parcels | `backend/local_parcels/` | **no** (private) |
| Host settings (port, domain, data paths) | `.env` (from `.env.example`) | no; optional |

Without the parcel data the app still runs (live prediction, Draw AOI,
uploads and Statistics all work), but the Parcel Browser and Live
Validation pages show fewer or no regions.

## Run locally

```bash
cp backend/.env.example backend/.env      # fill in CDSE_SH_CLIENT_ID / _SECRET
cp .env.example .env                      # optional; e.g. HTTP_PORT=8080 if 80 is taken
./deploy/deploy.sh --no-pull              # or: docker compose up -d --build
```

Open `http://localhost:<HTTP_PORT>`. Stop with `docker compose down`.

## Deploy to EC2

### 1. Launch the instance

- **AMI:** Amazon Linux 2023 or Ubuntu 24.04, x86_64.
- **Type:** `t3.small` (2 GB) is enough to run it; the backend measured about 210 MB idle. The heavy part is building the image (torch and the GDAL base), so the bootstrap script adds 2 GB of swap. Choose `t3.medium` if you want faster builds or expect concurrent users.
- **Disk:** at least 20 GB gp3. The backend image is about 3 GB, and rebuilds briefly need room for two copies.
- **Security group:** inbound 22 from your IP only, plus 80 and 443 from anywhere.
- **Elastic IP:** attach one so the address and DNS survive restarts.

### 2. Install Docker

```bash
ssh ec2-user@<ip>                         # ubuntu@<ip> on Ubuntu
sudo dnf install -y git || sudo apt-get install -y git
git clone https://github.com/parallaxx-dev/CropClassifier.git
cd CropClassifier
sudo ./deploy/bootstrap-ec2.sh
exit                                      # log back in for the docker group
```

### 3. Provide secrets and data

From your machine, copy over the files that aren't in git:

```bash
scp backend/.env ec2-user@<ip>:CropClassifier/backend/.env
rsync -avz backend/training/eurocrops_output/multi_country/checkpoints/*_sampled_geoms.geojson \
  ec2-user@<ip>:CropClassifier/backend/training/eurocrops_output/multi_country/checkpoints/
rsync -avz backend/local_parcels/ ec2-user@<ip>:CropClassifier/backend/local_parcels/
```

(Create the target directories with `mkdir -p` first if rsync complains.)

### 4. Deploy

```bash
ssh ec2-user@<ip>
cd CropClassifier && ./deploy/deploy.sh
```

The app is now served at `http://<ip>/`.

### 5. HTTPS (recommended)

1. Point a DNS A record (e.g. `crops.example.com`) at the Elastic IP.
2. On the host, add `DOMAIN=crops.example.com` to `.env`.
3. Run `./deploy/deploy.sh`. It detects `DOMAIN`, starts Caddy on ports 80 and 443, and gets the certificate automatically.

## Continuous deployment

`.github/workflows/cd.yml` SSHes into the instance on every push to `main`
and runs `deploy/deploy.sh`. It stays skipped until you configure it under
**Settings → Secrets and variables → Actions**:

- **Variables:** `EC2_HOST` (Elastic IP or DNS), `EC2_USER` (`ec2-user` or `ubuntu`), `EC2_APP_DIR` (e.g. `/home/ec2-user/CropClassifier`)
- **Secret:** `EC2_SSH_KEY`, a private key whose public half is in the instance's `~/.ssh/authorized_keys`. Use a dedicated deploy key, not your personal one.

The checkout on the host must be able to `git pull` without prompting.
Public repos work out of the box; for a private repo, add a read-only
deploy key on the host.

## Day-to-day operations

```bash
docker compose ps                         # status
docker compose logs -f backend            # live logs (rotated at 3×10 MB per container)
./deploy/deploy.sh                        # update to latest main
docker compose restart backend            # restart without rebuilding
docker compose down                       # stop everything
```

If you use the HTTPS overlay, add `-f docker-compose.yml -f docker-compose.https.yml`
to manual `docker compose` commands (`deploy.sh` does this for you).

Containers restart automatically on crash and on reboot
(`restart: unless-stopped`, with the Docker service enabled at boot).

## Leftover AWS resources from the earlier ECS plan

The earlier setup pushed to ECR (`academics/cropclassifier`, ap-south-1)
and created a backend ALB. Neither is used any more. Delete the ALB
(it bills hourly) and, if you like, the ECR images.
