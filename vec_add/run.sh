#!/usr/bin/env bash

set -e

mkdir -p build
nvcc -o build/vec_add_cpu vec_add_cpu.c
nvcc -o build/vec_add_gpu vec_add_gpu.cu

mkdir -p report
nsys profile \
    --cuda-memory-usage=true \
    --cuda-um-cpu-page-faults=true \
    --cuda-um-gpu-page-faults=true \
    --cudabacktrace=all \
    --event-sample=system-wide \
    --os-events=0,1,2,3,4,5,6,7,8 \
    --gpuctxsw=true \
    --trace=osrt,cuda \
    --output=report/vec_add_cpu.nsys-rep \
    --force-overwrite=true \
    build/vec_add_cpu

nsys stats \
    --report=osrt_sum \
    --report=cuda_api_sum \
    --report=cuda_kern_exec_sum \
    --report=cuda_gpu_trace \
    --report=cuda_gpu_kern_sum \
    --report=cuda_gpu_mem_time_sum \
    --report=cuda_gpu_mem_size_sum \
    --report=um_sum \
    --report=um_total_sum \
    --report=um_cpu_page_faults_sum \
    --report=um_gpu_page_faults_sum \
    --force-export=true \
    report/vec_add_cpu.nsys-rep

nsys profile \
    --cuda-memory-usage=true \
    --cuda-um-cpu-page-faults=true \
    --cuda-um-gpu-page-faults=true \
    --cudabacktrace=all \
    --event-sample=system-wide \
    --os-events=0,1,2,3,4,5,6,7,8 \
    --gpuctxsw=true \
    --trace=osrt,cuda \
    --output=report/vec_add_gpu.nsys-rep \
    --force-overwrite=true \
    build/vec_add_gpu

nsys stats \
    --report=osrt_sum \
    --report=cuda_api_sum \
    --report=cuda_kern_exec_sum \
    --report=cuda_gpu_trace \
    --report=cuda_gpu_kern_sum \
    --report=cuda_gpu_mem_time_sum \
    --report=cuda_gpu_mem_size_sum \
    --report=um_sum \
    --report=um_total_sum \
    --report=um_cpu_page_faults_sum \
    --report=um_gpu_page_faults_sum \
    --force-export=true \
    report/vec_add_gpu.nsys-rep