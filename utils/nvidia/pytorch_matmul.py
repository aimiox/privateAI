#
# aimiox.com - 2024
# Matrix multiplication benchmark test code
# usage: python3 pytorch_matmul.py
# example output (RTX 3050):
#      FP32 Time: 0.00039 seconds, Performance: 5448.84 GFLOPS with amp False
#      FP32 Time: 0.00021 seconds, Performance: 10310.60 GFLOPS with amp True
#      FP16 Time: 0.00014 seconds, Performance: 15062.40 GFLOPS with amp False
#      FP16 Time: 0.00014 seconds, Performance: 15062.80 GFLOPS with amp True
#

import torch
import time
from torch.cuda.amp import autocast

# Matrix dimensions
N = 1024

# Function to measure GFLOPS
def measure_gflops(precision, dtype, use_amp=False, iter=10000):
    A = torch.randn(N, N, device='cuda', dtype=dtype)
    B = torch.randn(N, N, device='cuda', dtype=dtype)
    C = torch.zeros(N, N, device='cuda', dtype=dtype)

    # Warm-up
    if use_amp:
        for _ in range(10):
            with autocast():
                torch.matmul(A, B)
    else:
        for _ in range(10):
            torch.matmul(A, B)

    # Measure time
    start = time.time()
    if use_amp:
        for _ in range(iter):
            with autocast():
                torch.matmul(A, B)
    else:
        for _ in range(iter):
            torch.matmul(A, B)
    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / iter
    gflops = (2 * N ** 3) / (avg_time * 1e9)

    print(f"{precision} Time: {avg_time:.5f} seconds, Performance: {gflops:.2f} GFLOPS with amp {use_amp}")

# Ensure the matrix size is a multiple of 8
N = (N // 8) * 8

# FP32 Benchmark
measure_gflops('FP32', torch.float32, use_amp=False)
measure_gflops('FP32', torch.float32, use_amp=True)

# FP16 Benchmark
measure_gflops('FP16', torch.float16, use_amp=False)
measure_gflops('FP16', torch.float16, use_amp=True)

