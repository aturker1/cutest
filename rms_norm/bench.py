import torch
from triton.testing import do_bench
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from naive import (
    rms_norm_kernel,
    rms_norm_fwd,
    rms_norm_fwd_tvm_ffi,
    ASSUMED_ALIGN_BYTES,
)
from quack.rmsnorm import rmsnorm_fwd


def bench(M, N=128, dtype=torch.bfloat16):
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)
    eps = 1e-6

    with torch.inference_mode():
        # Reference (torch)
        ref = torch.nn.functional.rms_norm(x, (N,), weight=w, eps=eps)
        torch_time = do_bench(
            lambda: torch.nn.functional.rms_norm(x, (N,), weight=w, eps=eps)
        )

        # Compile torch kernel
        compiled_torch = torch.compile(
            torch.nn.functional.rms_norm, dynamic=True, mode="reduce-overhead"
        )
        compiled_torch(x, (N,), weight=w, eps=eps)
        compiled_torch_time = do_bench(
            lambda: compiled_torch(x, (N,), weight=w, eps=eps)
        )

    # Quack kernel
    quack_out,*_ = rmsnorm_fwd(x, w, eps=eps)
    quack_time = do_bench(lambda: rmsnorm_fwd(x, w, eps=eps))
    torch.testing.assert_close(quack_out, ref)

    # CuTe kernel (wrapper): includes output allocation + DLPack conversions in the hot path
    rms_norm_fwd(x, w, eps)
    cute_wrapper_time = do_bench(lambda: rms_norm_fwd(x, w, eps))
    y = rms_norm_fwd(x, w, eps)
    torch.testing.assert_close(y, ref)

    # CuTe kernel (direct): preconvert + precompile once (matches "direct call" numbers)
    y_direct = torch.empty_like(x)
    x_cute = from_dlpack(x, assumed_align=ASSUMED_ALIGN_BYTES)
    y_cute = from_dlpack(y_direct, assumed_align=ASSUMED_ALIGN_BYTES)
    w_cute = from_dlpack(w, assumed_align=ASSUMED_ALIGN_BYTES)
    compiled_cute = cute.compile(rms_norm_kernel, x_cute, y_cute, w_cute, eps)
    compiled_cute(x_cute, y_cute, w_cute)
    cute_direct_time = do_bench(lambda: compiled_cute(x_cute, y_cute, w_cute))
    torch.testing.assert_close(y_direct, ref)

    # CuTe kernel (tvm_ffi): compiled once with symbolic M
    y_tvm = rms_norm_fwd_tvm_ffi(x, w, eps)
    cute_tvm_time = do_bench(lambda: rms_norm_fwd_tvm_ffi(x, w, eps))
    torch.testing.assert_close(y_tvm, ref)

    bwd = lambda m, n, time: M * N * dtype.itemsize * 2 / time / 1e6

    print(
        f"M={M:7d} | "
        f"torch: {bwd(M, N, torch_time):4.1f} GB/s | "
        f"compiled_torch: {bwd(M, N, compiled_torch_time):4.1f} GB/s | "
        f"quack: {bwd(M, N, quack_time):4.1f} GB/s | "
        f"tvm_ffi(sym): {bwd(M, N, cute_tvm_time):4.1f} GB/s | "
        f"direct: {bwd(M, N, cute_direct_time):4.1f} GB/s | "
        f"wrapper: {bwd(M, N, cute_wrapper_time):4.1f} GB/s"
    )


if __name__ == "__main__":
    print("N=128, dtype=bfloat16")
    print("-" * 85)
    M = 1024

    for _ in range(14):
        bench(M)
        M *= 2
