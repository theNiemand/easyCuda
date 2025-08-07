#include "common.h"
#include "vector_add.h"
#include "gemm_sliced_k.h"

PYBIND11_MODULE(_C, m)
{
    m.def("vector_add", vector_add);
    m.def("gemm_sliced_k", gemm_sliced_k);
}