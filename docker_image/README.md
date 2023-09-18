
# Docker Image

This directory contains a Dockerfile for building a Docker image that includes the Hailo-8 AI accelerator driver and runtime libraries, as well as the app for object detection.

## Requirements

- Docker installed
- Hailort can be downloaded from https://hailo.ai/
- Hailo PCi driver installed on your machine

## Building the Docker Image

To build the Docker image:
1. navigate to the `docker_image` directory 
2. move the " hailort_*_amd64.deb "
3. and run the following command:
```
docker build -t ubuntu22-decktop .
```

This will build the Docker image and tag it as `ubuntu22-decktop`.

## Running the Docker Image

To run the Docker image, use the following command:

```
docker run --device=/dev/video0 --device=/dev/hailo0 -it --env=DISPLAY --volume=/tmp/.X11-unix:/tmp/.X11-unix  ubuntu22-decktop /bin/bash
```

This will start a container from the `ubuntu22-decktop` image and open a command prompt inside the container.

## Usage

Once inside the container, you can run the Hailo Apps.

