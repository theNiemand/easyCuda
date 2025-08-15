import torch
import time
from typing import Callable, List

def snr_error(pred: torch.Tensor, real: torch.Tensor) -> float:
    """
    Compute SNR(Singal Noise Ratio) error between pred and real.

                    ∑((ele_pred - ele_real) ^2)    
    snr_error =  —————————————————————————————————
                        ∑(ele_real ^2)         

    Args:
        pred: predicted tensor
        real: real tensor

    Returns: snr_error
    """
    pred = torch.flatten(pred).float()
    real = torch.flatten(real).float()

    if pred.shape != real.shape:
        raise ValueError(f"Can not compute snr loss for tensors with different shape. ({pred.shape} and {real.shape})")

    noise_power = torch.pow(pred - real, 2).sum(dim=-1)
    signal_power = torch.pow(real, 2).sum(dim=-1)
    snr = (noise_power) / (signal_power + 1e-7)
    return snr.item()


def benchmark(Op: Callable, shape: List, flop: int, *args, **kwargs):
    """
    Benchmark the Op's tflops with the given flop.
    Benchmark the Op's throughput with the numel of given args and kwargs.

    Args:
        Op: the operator to benchmark
        shape: the shape of the input, just for print
        flop: the floating-point operation of the operator
        *args: the arguments of the operator
        **kwargs: the keyword arguments of the operator
    """
    # cal running average time
    total_time = 0
    times = 100

    # warm up
    warmup_times = 10
    for _ in range(warmup_times):
        Op(*args, **kwargs)

    torch.cuda.synchronize()
    for _ in range(times):
        start_time = time.perf_counter()

        Op(*args, **kwargs)
        torch.cuda.synchronize()

        end_time = time.perf_counter()
        total_time += end_time - start_time

    running_average_time = total_time / times

    # cal tflops
    Tera = 1e12
    tflops = flop / running_average_time / Tera

    # cal throughput
    numel = sum(arg.numel() for arg in args if isinstance(arg, torch.Tensor)) + \
            sum(kwarg_v.numel() for kwarg_v in kwargs.values() if isinstance(kwarg_v, torch.Tensor))

    Giga = 1e9
    throughput = numel / running_average_time / Giga

    # print in a good format in different lines
    print(f"Op: [{Op.__name__}], shape: {shape}\n")
    print(f"[average time]: {running_average_time * 1000 :2f}ms\n")
    print(f"[tflops]: {tflops :.2f}TFLOPS\n")
    print(f"[throughput]: {throughput :.2f}GB/s\n\n")