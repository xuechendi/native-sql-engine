#!/usr/bin/env bash

set -eu

TESTS=${1:-OFF}
BUILD_ARROW=${2:-ON}
STATIC_ARROW=${3:-OFF}
BUILD_PROTOBUF=${4:-ON}
ARROW_ROOT=${5:-/usr/local}

echo "CMAKE Arguments:"
echo "TESTS=${TESTS}"
echo "BUILD_ARROW=${BUILD_ARROW}"
echo "STATIC_ARROW=${STATIC_ARROW}"
echo "BUILD_PROTOBUF=${BUILD_PROTOBUF}"
echo "ARROW_ROOT=${ARROW_ROOT}"

CURRENT_DIR=$(cd "$(dirname "$BASH_SOURCE")"; pwd)
echo $CURRENT_DIR

cd ${CURRENT_DIR}
if [ -d build ]; then
    rm -r build
fi
mkdir build
cd build
cmake .. -DTESTS=${TESTS} -DBUILD_ARROW=${BUILD_ARROW} -DSTATIC_ARROW=${STATIC_ARROW} -DBUILD_PROTOBUF=${BUILD_PROTOBUF} -DARROW_ROOT=${ARROW_ROOT}
make

set +eu
