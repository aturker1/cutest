import torch
from add import vectorized_add_fwd_tvm_ffi
from triton.testing import do_bench
import pandas as pd


def torch_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b


def bench_add():
    shapes = [1024, 2048, 4096, 8192, 16384, 16384 + 8192]
    dtype = torch.bfloat16

    df = pd.DataFrame(
        columns=[
            "Shape",
            "CUTE FFI Timing",
            "Torch Timing",
            "CUTE FFI Band",
            "Torch Band",
        ]
    )

    for shape in shapes:
        a = torch.randn(shape, shape, device="cuda", dtype=dtype)
        b = torch.randn(shape, shape, device="cuda", dtype=dtype)
        torch_output = torch_add(a, b)
        torch.testing.assert_close(vectorized_add_fwd_tvm_ffi(a, b), torch_output)

        # Warmup
        for _ in range(10):
            vectorized_add_fwd_tvm_ffi(a, b)
            torch_add(a, b)

        # Benchmark
        cute_ffi_timing = do_bench(lambda: vectorized_add_fwd_tvm_ffi(a, b))
        torch_timing = do_bench(lambda: torch_add(a, b))
        cute_ffi_bandwidth = shape * shape * dtype.itemsize * 3 / cute_ffi_timing / 1e9
        torch_bandwidth = shape * shape * dtype.itemsize * 3 / torch_timing / 1e9

        df.loc[len(df)] = [
            shape,
            cute_ffi_timing,
            torch_timing,
            cute_ffi_bandwidth,
            torch_bandwidth,
        ]

    print(df)


if __name__ == "__main__":
    bench_add()
