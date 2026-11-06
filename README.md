# Sheriff Scanner

This is where my development progress will be shared for using machine learning and LIDAR to detect other FRC robots during a match. This is one stepping stone to one day having a fully AI strategised and driven match. It all starts with one thing: robot localization of not just your own robot (that's easy) but everyone around you in high speed matches.

## Installation

### 1. Pull the latest Docker image

Run this on an Internet-connected device:

```bash
docker pull --platform linux/arm64 ghcr.io/hjmcphail76/sheriffscanner/sheriff-scanner:latest
```

### 2. Push the image to the Orange Pi over SSH

Replace `<pi-ip>` with your Orange Pi's IP address:

```bash
docker save ghcr.io/hjmcphail76/sheriffscanner/sheriff-scanner:latest | gzip | ssh orangepi@<pi-ip> "gunzip | docker load"
```

### 3. Replace the old container with the new deployment

Run this on the Orange Pi. It removes the old container (if any) and starts a new runner:

```bash
docker rm -f scanner-runner 2>/dev/null || true
docker run -d --name scanner-runner --restart unless-stopped \
  --device /dev/ttyUSB0:/dev/ttyUSB0 -p 8080:8080 -e SDL_VIDEODRIVER=dummy \
  ghcr.io/hjmcphail76/sheriffscanner/sheriff-scanner:latest
```