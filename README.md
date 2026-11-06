# SheriffScanner

This is where my development progress will be shared for using machine learning and LIDAR to detect other FRC robots during a match. This is one stepping stone to one day having a fully AI-strategised and driven match. It all starts with one thing: robot localization of not just your own robot (that's easy) but everyone around you in high speed matches.

The scanner reads an [RPLidar C1]([url](https://www.amazon.com/dp/B0DDKZV8HZ?lv=shuf&channelId=500&plpRedirect=mhFallback)), clusters the points into targets with DBSCAN, and publishes their poses to NetworkTables (`LidarProcessor/poses`).



## One-time setup on the Orange Pi

Install Docker and make sure it starts at boot:

```bash
sudo systemctl enable --now docker
```

Find the LIDAR's serial device (Note: the driver comes preinstalled on the Orange Pi 5 Debian Bookworm image):

```bash
ls /dev/ttyUSB* /dev/ttyACM*
```

Use that path for `--device` below (this README assumes `/dev/ttyUSB0`).

## Running the container (starts on every boot after this initial command)

```bash
docker run -d \
  --name scanner-runner \
  --restart unless-stopped \
  --device /dev/ttyUSB0:/dev/ttyUSB0 \
  -p 8080:8080 \
  ghcr.io/hjmcphail76/sheriffscanner/sheriff-scanner:latest
```

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

## Optional: automatic updates

If the Pi has a reliable internet connection, Watchtower can watch GHCR and swap the container when a new image appears. Run `docker run` once first so the container exists:

```bash
docker run -d --name watchtower --restart unless-stopped \
  -v /var/run/docker.sock:/var/run/docker.sock \
  containrrr/watchtower --interval 60 scanner-runner
```


## Cleanup

Remove all containers (this stops the scanner until you start a new one):

```bash
docker rm -f $(docker ps -aq)
```

Remove unused images to free disk space:

```bash
docker image prune -a
```
