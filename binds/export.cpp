#include "common.h"
#include "vector_add.h"

PYBIND11_MODULE(_C, m)
{
    m.def("vector_add", vector_add);
}