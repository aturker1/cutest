import torch
from triton.testing import do_bench
from add import vectorized_add_fwd_tvm_ffi, vectorized_add_wrapper


def bench_ms(fn, rep=300):
    torch.cuda.synchronize()
    return do_bench(
        lambda: (fn(), torch.cuda.synchronize()), rep=rep
    )  # Sync after fn to able to measure host overhead


if __name__ == "__main__":
    shape_start = 128

    for step in range(0, 9):
        shape = shape_start * 2**step
        a = torch.randn(shape, shape, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(shape, shape, device="cuda", dtype=torch.bfloat16)

        torch.testing.assert_close(vectorized_add_fwd_tvm_ffi(a, b), a + b)
        torch.testing.assert_close(vectorized_add_wrapper(a, b), a + b)

        bench_a = torch.randn(shape, shape, device="cuda", dtype=torch.bfloat16)
        bench_b = torch.randn(shape, shape, device="cuda", dtype=torch.bfloat16)

        ffi_ms = bench_ms(lambda: vectorized_add_fwd_tvm_ffi(bench_a, bench_b))
        base_ms = bench_ms(lambda: vectorized_add_wrapper(bench_a, bench_b))
        torch_ms = bench_ms(lambda: bench_a + bench_b)
        bw = lambda ms: shape * shape * a.element_size() * 3 / ms / 1e9  # noqa: E731

        print(
            f"{shape}x{shape}: "
            f"CuTe DSL={base_ms:.6f} ms | "
            f"Cute DSL FFI={ffi_ms:.6f}ms | "
            f"Torch={torch_ms:.6f}ms | "
            f"CuTe DSL BW={bw(base_ms):.4f} TB/s | "
            f"CuTE DSL FFI BW={bw(ffi_ms):.4f} TB/s | "
            f"Torch BW={bw(torch_ms):.4f} TB/s"
        )
