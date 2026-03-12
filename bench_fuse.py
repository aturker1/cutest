"""Benchmark: CuTe DSL fused kernel vs torch.compile vs PyTorch eager."""

import torch
from triton.testing import do_bench

from cutest import Tensor
from cutest.compiler import compile as cutest_compile


def torch_eager(a, b, c, d):
    return (a + b) * c - d


torch_compiled = torch.compile(torch_eager)


def bench_fuse():
    shapes = [256, 512, 1024, 2048, 4096, 8192]
    dtype = torch.float16

    # Header
    print(
        f"{'Shape':>10} | {'CuTe Fused':>12} | {'torch.compile':>14} | "
        f"{'Torch Eager':>12} | {'CuTe BW':>10} | {'Compile BW':>10} | "
        f"{'Eager BW':>10}"
    )
    print("-" * 100)

    for shape in shapes:
        a = torch.randn(shape, shape, device="cuda", dtype=dtype)
        b = torch.randn(shape, shape, device="cuda", dtype=dtype)
        c = torch.randn(shape, shape, device="cuda", dtype=dtype)
        d = torch.randn(shape, shape, device="cuda", dtype=dtype)

        # Build graph and compile cutest kernel once
        A, B, C, D = Tensor(a), Tensor(b), Tensor(c), Tensor(d)
        E = (A + B) * C - D
        fused_fn = cutest_compile(E)

        # Verify correctness
        expected = torch_eager(a, b, c, d)
        result = fused_fn(a, b, c, d)
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

        # Warmup torch.compile
        for _ in range(5):
            torch_compiled(a, b, c, d)

        # Benchmark (kernel-only, no graph overhead)
        cute_ms = do_bench(lambda: fused_fn(a, b, c, d))
        compile_ms = do_bench(lambda: torch_compiled(a, b, c, d))
        eager_ms = do_bench(lambda: torch_eager(a, b, c, d))

        # Bandwidth: 4 reads + 1 write = 5 tensors
        nbytes = shape * shape * dtype.itemsize * 5
        cute_bw = nbytes / cute_ms / 1e9
        compile_bw = nbytes / compile_ms / 1e9
        eager_bw = nbytes / eager_ms / 1e9

        print(
            f"{shape:>5}x{shape:<4} | {cute_ms:>10.4f}ms | {compile_ms:>12.4f}ms | "
            f"{eager_ms:>10.4f}ms | {cute_bw:>8.2f}TB/s | {compile_bw:>8.2f}TB/s | "
            f"{eager_bw:>8.2f}TB/s"
        )


if __name__ == "__main__":
    bench_fuse()
