#!/usr/bin/env bash

set -e

mkdir -p build
nvcc -o build/vec_add_cpu vec_add_cpu.c
nvcc -o build/vec_add_gpu vec_add_gpu.cu

build/vec_add_cpu
build/vec_add_gpu
