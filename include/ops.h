#pragma once

#include  "common.h"

torch::Tensor vector_add(
    const torch::Tensor& x,
    const torch::Tensor& y
);

torch::Tensor gemm_sliced_k_sm80_naive(
    const torch::Tensor& A,
    const torch::Tensor& B
);


torch::Tensor gemm_sliced_k_sm80_optim(
    const torch::Tensor& A,
    const torch::Tensor& B
);