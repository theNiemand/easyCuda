#pragma once

#include <cuda.h>             // CUDA运行时基础头文件
#include <cuda_runtime.h>     // CUDA运行时API（如cudaMalloc, cudaMemcpy）
#include <cuda_fp16.h>        // fp16支持
#include <cuda_bf16.h>        // bf16支持
#include <torch/extension.h>  // PyTorch核心扩展头文件（包含所有基础功能）
#include <ATen/cuda/CUDAContext.h>

