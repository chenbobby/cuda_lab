#!/usr/bin/env bash

set -e

mkdir -p build
nvcc -o build/vec_add --run main.cu
