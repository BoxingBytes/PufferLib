#!/bin/bash
# Usage (from anywhere): ocean/common_harvest/debug_play.sh [--env.key=value ...]
set -e
cd "$(dirname "$0")/../.."
RAYLIB=raylib-5.5_linux_amd64
clang -g -O0 -Wall -Wno-unused-function \
    -I. -Isrc -Iocean/common_harvest -Ivendor -I$RAYLIB/include \
    ocean/common_harvest/debug_play.c $RAYLIB/lib/libraylib.a \
    -lm -lpthread -ldl -lGL -DPLATFORM_DESKTOP \
    -o debug_harvest
./debug_harvest "$@"
