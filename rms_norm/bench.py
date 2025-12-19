import torch
from triton.testing import do_bench
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from naive import rms_norm_kernel
from quack.rmsnorm import rmsnorm_fwd


def bench(M, N=128, dtype=torch.bfloat16):
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)
    eps = 1e-6

    # Reference (torch)
    ref = torch.nn.functional.rms_norm(x, (N,), weight=w, eps=eps)
    torch_time = do_bench(
        lambda: torch.nn.functional.rms_norm(x, (N,), weight=w, eps=eps)
    )

    # Compile torch kernel
    compiled_torch = torch.compile(torch.nn.functional.rms_norm, dynamic=True)
    compiled_torch(x, (N,), weight=w, eps=eps)
    compiled_torch_time = do_bench(lambda: compiled_torch(x, (N,), weight=w, eps=eps))

    # Quack kernel
    quack_y, quack_residual, quack_rstd = rmsnorm_fwd(x, w, eps=eps)
    quack_time = do_bench(lambda: rmsnorm_fwd(x, w, eps=eps))

    y = torch.zeros_like(x)

    # TODO: Keep this for simplicity, but it should be (1, N)
    scale_2d = torch.ones((16, N), dtype=dtype, device="cuda") * w

    x_cute = from_dlpack(x, assumed_align=16)
    y_cute = from_dlpack(y, assumed_align=16)
    scale_cute = from_dlpack(scale_2d, assumed_align=16)

    kernel = cute.compile(rms_norm_kernel, x_cute, y_cute, scale_cute, eps)
    kernel(x_cute, y_cute, scale_cute)

    # CuTe kernel
    cute_time = do_bench(lambda: kernel(x_cute, y_cute, scale_cute))
    cute_mae = (y - ref).abs().mean().item()

    speedup = torch_time / cute_time
    cute_bandwidth = M * N * dtype.itemsize * 2 / cute_time / 1e6
    torch_bandwidth = M * N * dtype.itemsize * 2 / torch_time / 1e6
    compiled_torch_bandwidth = M * N * dtype.itemsize * 2 / compiled_torch_time / 1e6
    quack_bandwidth = M * N * dtype.itemsize * 2 / quack_time / 1e6
    print(
        f"M={M:6d} | Torch: {torch_bandwidth:.4f} GB/s | Compiled Torch: {compiled_torch_bandwidth:.4f} GB/s | Quack: {quack_bandwidth:.4f} GB/s | CuTe: {cute_bandwidth:.4f} GB/s | Speedup: {speedup:.2f}x | MAE: {cute_mae:.2e}"
    )


if __name__ == "__main__":
    print("N=128, dtype=bfloat16")
    print("-" * 85)
    M = 1024

    for _ in range(10):
        bench(M)
        M *= 2
