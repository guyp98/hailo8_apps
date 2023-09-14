#!/bin/bash

# Export the Docker image to a tar file
docker save ubuntu22-decktop > ubuntu22-decktop.tar

# Compress the tar file using gzip
gzip ubuntu22-decktop.tar