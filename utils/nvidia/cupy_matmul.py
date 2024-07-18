#
# aimiox.com - 2024
# Matrix multiplication benchmark test code
# usage: python3 cupy_matmul.py
# example output (RTX 3050):
#      FP32 Time: 0.00039 seconds, Performance: 5443.81 GFLOPS
#      FP16 Time: 0.00014 seconds, Performance: 14965.46 GFLOPS
#

import cupy as cp
import time

# Matrix dimensions
N = 1024

# Function to measure GFLOPS
def measure_gflops(precision, dtype, iter=10000):
    A = cp.random.randn(N, N).astype(dtype)
    B = cp.random.randn(N, N).astype(dtype)
    C = cp.zeros((N, N), dtype=dtype)

    # Warm-up
    for _ in range(10):
        cp.matmul(A, B)

    # Measure time
    start = time.time()
    for _ in range(iter):
        cp.matmul(A, B)
    cp.cuda.Stream.null.synchronize()
    end = time.time()

    avg_time = (end - start) / iter
    gflops = (2 * N ** 3) / (avg_time * 1e9)

    print(f"{precision} Time: {avg_time:.5f} seconds, Performance: {gflops:.2f} GFLOPS")

# Ensure the matrix size is a multiple of 8
N = (N // 8) * 8

# FP32 Benchmark
measure_gflops('FP32', cp.float32)

# FP16 Benchmark
measure_gflops('FP16', cp.float16)
