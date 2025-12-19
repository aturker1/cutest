import torch
from triton.testing import do_bench
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from naive import rms_norm_kernel


def bench(M, N=128, dtype=torch.bfloat16):
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)
    eps = 1e-6

    # Reference (torch)
    ref = torch.nn.functional.rms_norm(x, (N,), weight=w, eps=eps)
    torch_time = do_bench(
        lambda: torch.nn.functional.rms_norm(x, (N,), weight=w, eps=eps)
    )

    # CuTe naive
    y = torch.zeros_like(x)
    scale_2d = torch.ones((2, N), dtype=dtype, device="cuda") * w
    x_cute = from_dlpack(x, assumed_align=16)
    y_cute = from_dlpack(y, assumed_align=16)
    scale_cute = from_dlpack(scale_2d, assumed_align=16)
    kernel = cute.compile(rms_norm_kernel, x_cute, y_cute, scale_cute, eps)
    kernel(x_cute, y_cute, scale_cute)
    cute_time = do_bench(lambda: kernel(x_cute, y_cute, scale_cute))
    cute_mae = (y - ref).abs().mean().item()

    speedup = torch_time / cute_time
    print(
        f"M={M:6d} | Torch: {torch_time:.4f}ms | CuTe: {cute_time:.4f}ms | Speedup: {speedup:.2f}x | MAE: {cute_mae:.2e}"
    )


if __name__ == "__main__":
    print("N=128, dtype=bfloat16")
    print("-" * 85)
    M = 1024
    while M <= 262144:
        bench(M)
        M *= 2
