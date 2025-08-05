import importlib
import os
import ctypes
import torch

# -------- 预加载 libtorch_python.so --------
_libtorch_py = os.path.join(os.path.dirname(torch.__file__),
                            "lib", "libtorch_python.so")
ctypes.CDLL(_libtorch_py, mode=ctypes.RTLD_GLOBAL)
# ------------------------------------------


PKG = "easyCuda"
ops = importlib.import_module(f"{PKG}._C")