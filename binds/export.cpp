#include "common.h"
#include "ops.h"

PYBIND11_MODULE(_C, m)
{
    m.def("vector_add", vector_add);
    m.def("gemm_sliced_k_sm80_naive", gemm_sliced_k_sm80_naive);
    m.def("gemm_sliced_k_sm80_optim", gemm_sliced_k_sm80_optim);
}