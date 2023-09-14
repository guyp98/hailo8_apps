#!/bin/bash

# Load the Docker image from the tar file
docker load < ubuntu22-decktop.tar.gz

# Run a container from the image
docker run --device=/dev/video0 --device=/dev/hailo0 -it \
        --env=DISPLAY --volume=/tmp/.X11-unix:/tmp/.X11-unix  ubuntu22-decktop /bin/bash

# Attach to the container
# docker attach ubuntu22-decktop