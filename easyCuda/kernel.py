import os
import torch
from . import ops

def vector_add(x:torch.Tensor, y:torch.Tensor):
    return ops.vector_add(x, y)