import os
import torch
from . import ops

def vector_add(x:torch.Tensor, y:torch.Tensor):
    return ops.vector_add(x, y)

def gemm_sliced_k_sm80_naive(A:torch.Tensor, B:torch.Tensor):
    return ops.gemm_sliced_k_sm80_naive(A, B)

def gemm_sliced_k_sm80_optim(A:torch.Tensor, B:torch.Tensor):
    return ops.gemm_sliced_k_sm80_optim(A, B)