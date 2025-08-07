#pragma once

#include  "common.h"

torch::Tensor gemm_sliced_k(
    const torch::Tensor& A,
    const torch::Tensor& B
    );