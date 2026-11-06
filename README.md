# SheriffScanner

This is where my development progress will be shared for using machine learning and LIDAR to detect other FRC robots during a match. This is one stepping stone to one day having a fully AI-strategised and driven match. It all starts with one thing: robot localization of not just your own robot (that's easy) but everyone around you in high speed matches.

The scanner reads an RPLidar C1, clusters the points into targets with DBSCAN, and publishes their poses to NetworkTables (`LidarProcessor/poses`).

## How deployment works

1. Pushing to `main` triggers the GitHub Action in `.github/workflows/`.
2. The Action builds an **arm64** Docker image (for the Orange Pi) from `./sheriff_scanner`.
3. The image is pushed to GitHub Container Registry (GHCR) and also uploaded as a downloadable tarball artifact (`scanner-image`, kept 14 days).
4. The Orange Pi gets the new image either by pulling it from GHCR (needs internet) or by loading the tarball (no internet needed on the Pi).

Image name:

```
ghcr.io/hjmcphail76/sheriffscanner/sheriff-scanner:latest
```

## One-time setup on the Orange Pi

Install Docker and make sure it starts at boot:

```bash
sudo systemctl enable --now docker
```

Find the LIDAR's serial device:

```bash
ls /dev/ttyUSB* /dev/ttyACM*
```

Use that path for `--device` below (this README assumes `/dev/ttyUSB0`). Make sure `LIDAR_PORT` in `main.py` matches it. It must not be a Windows port name like `COM3`.

If the GHCR package is private, either make it public (GitHub → your profile → Packages → `sheriff-scanner` → Package settings → Change visibility) or log in on the Pi with a personal access token that has `read:packages`:

```bash
echo <TOKEN> | docker login ghcr.io -u hjmcphail76 --password-stdin
```

## Running the container (starts on every boot)

`--restart unless-stopped` is what makes the container start on boot. Docker itself only needs to be running locally, so this works with no internet.

```bash
docker run -d \
  --name scanner-runner \
  --restart unless-stopped \
  --device /dev/ttyUSB0:/dev/ttyUSB0 \
  -p 8080:8080 \
  -e SDL_VIDEODRIVER=dummy \
  ghcr.io/hjmcphail76/sheriffscanner/sheriff-scanner:latest
```

Notes:

- `SDL_VIDEODRIVER=dummy` stops pygame from crashing in a headless container. The plot window goes nowhere.
- `nt_interface.py` connects to NetworkTables at `127.0.0.1`. Inside a bridge-networked container that is the container itself, not the Pi. If the NT server runs on the Pi or the robot, either run with `--network host` (and drop `-p 8080:8080`) or change the server address in `nt_interface.py` to the real IP.

Check that it's healthy:

```bash
docker ps
docker logs --tail 30 scanner-runner
docker inspect -f 'policy={{.HostConfig.RestartPolicy.Name}} restarts={{.RestartCount}}' scanner-runner
```

A restart count that keeps climbing means it is crash-looping; the logs will say why.

## Updating to a new build

### Option 1: Pi has internet (pull from GHCR)

Save as `~/update.sh` on the Pi and run `chmod +x ~/update.sh`:

```bash
#!/bin/bash
set -e
IMAGE=ghcr.io/hjmcphail76/sheriffscanner/sheriff-scanner:latest

docker pull $IMAGE
docker rm -f scanner-runner 2>/dev/null || true
docker run -d \
  --name scanner-runner \
  --restart unless-stopped \
  --device /dev/ttyUSB0:/dev/ttyUSB0 \
  -p 8080:8080 \
  -e SDL_VIDEODRIVER=dummy \
  $IMAGE
docker image prune -f
```

Run `~/update.sh` after the Action finishes (watch the Actions tab). Because of `set -e`, if the pull fails (no internet) the script stops before touching the running container.

### Option 2: Pi has no internet (load a tarball, no Docker needed on your laptop)

1. Open the finished run in the repo's **Actions** tab and download the `scanner-image` artifact. Unzip it to get `scanner.tar.gz`.
2. Copy it to the Pi, with `scp` or a USB stick:

   ```bash
   scp scanner.tar.gz orangepi@<pi-ip>:~
   ```

3. On the Pi, load the image and restart the container:

   ```bash
   gunzip -c scanner.tar.gz | docker load
   docker rm -f scanner-runner 2>/dev/null || true
   docker run -d \
     --name scanner-runner \
     --restart unless-stopped \
     --device /dev/ttyUSB0:/dev/ttyUSB0 \
     -p 8080:8080 \
     -e SDL_VIDEODRIVER=dummy \
     ghcr.io/hjmcphail76/sheriffscanner/sheriff-scanner:latest
   ```

### Option 3: Build directly on the Pi

Copy the `sheriff_scanner` folder to the Pi and build there (needs internet for `pip install`):

```bash
cd sheriff_scanner
docker build -t sheriff_scanner .
```

Then run it with the same `docker run` flags as above, using `sheriff_scanner` as the image name.

## Optional: automatic updates

If the Pi has a reliable internet connection, Watchtower can watch GHCR and swap the container when a new image appears. Run `docker run` once first so the container exists:

```bash
docker run -d --name watchtower --restart unless-stopped \
  -v /var/run/docker.sock:/var/run/docker.sock \
  containrrr/watchtower --interval 60 scanner-runner
```

Before a competition, update manually and test rather than relying on this. A broken push would otherwise replace a working version automatically.

## Cleanup

Remove all containers (this stops the scanner until you start a new one):

```bash
docker rm -f $(docker ps -aq)
```

Remove unused images to free disk space:

```bash
docker image prune -a
```
