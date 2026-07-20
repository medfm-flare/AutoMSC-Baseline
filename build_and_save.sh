#!/bin/sh
# Build the FLARE-AutoMSC submission image and save it to <team>.tar.gz.
#
# Usage:  sh build_and_save.sh [teamname]
# Before running: place your trained model under model_weights/ (see model_weights/README.md).
set -e

TEAM=${1:-teamname}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

docker build -t "${TEAM}:latest" -f "${SCRIPT_DIR}/docker/Dockerfile" "${SCRIPT_DIR}"
docker save "${TEAM}:latest" | gzip -c > "${SCRIPT_DIR}/${TEAM}.tar.gz"

echo "Saved ${SCRIPT_DIR}/${TEAM}.tar.gz"
echo "Test with:"
echo "  docker load -i ${TEAM}.tar.gz"
echo "  docker container run --gpus \"device=1\" -m 28G --name ${TEAM} --rm \\"
echo "    -v \$PWD/FLARE_Test/:/workspace/inputs/ \\"
echo "    -v \$PWD/${TEAM}_outputs/:/workspace/outputs/ \\"
echo "    ${TEAM}:latest /bin/bash -c \"sh predict.sh\""
