import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from add import naive_add_kernel, vectorized_add_kernel, tv_layout_add
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
            "CUTE Timing",
            "CUTE Vec Timing",
            "CUTE TV Timing",
            "Torch Timing",
            "CUTE Band",
            "CUTE Vec Band",
            "CUTE TV Band",
            "Torch Band",
        ]
    )

    for shape in shapes:
        a = torch.randn(shape, shape, device="cuda", dtype=dtype)
        b = torch.randn(shape, shape, device="cuda", dtype=dtype)
        c = torch.zeros(shape, shape, device="cuda", dtype=dtype)

        a_dlpack = from_dlpack(a, assumed_align=16)
        b_dlpack = from_dlpack(b, assumed_align=16)
        c_dlpack = from_dlpack(c, assumed_align=16)

        compiled_add = cute.compile(naive_add_kernel, a_dlpack, b_dlpack, c_dlpack)
        compiled_vectorized_add = cute.compile(
            vectorized_add_kernel, a_dlpack, b_dlpack, c_dlpack
        )
        compiled_tv_layout_add = cute.compile(
            tv_layout_add, a_dlpack, b_dlpack, c_dlpack
        )
        # Validate
        compiled_add(a_dlpack, b_dlpack, c_dlpack)
        torch_output = torch_add(a, b)
        torch.testing.assert_close(c, torch_output)

        compiled_vectorized_add(a_dlpack, b_dlpack, c_dlpack)
        torch.testing.assert_close(c, torch_output)
        compiled_tv_layout_add(a_dlpack, b_dlpack, c_dlpack)
        torch.testing.assert_close(c, torch_output)

        # Warmup
        for _ in range(10):
            compiled_add(a_dlpack, b_dlpack, c_dlpack)
            compiled_vectorized_add(a_dlpack, b_dlpack, c_dlpack)
            compiled_tv_layout_add(a_dlpack, b_dlpack, c_dlpack)
            torch_add(a, b)

        # Benchmark
        cute_timing = do_bench(lambda: compiled_add(a_dlpack, b_dlpack, c_dlpack))
        torch_timing = do_bench(lambda: torch_add(a, b))
        cute_vectorized_timing = do_bench(
            lambda: compiled_vectorized_add(a_dlpack, b_dlpack, c_dlpack)
        )
        cute_tv_layout_timing = do_bench(
            lambda: compiled_tv_layout_add(a_dlpack, b_dlpack, c_dlpack)
        )
        cute_bandwidth = shape * shape * dtype.itemsize * 3 / cute_timing / 1e9
        cute_vectorized_bandwidth = (
            shape * shape * dtype.itemsize * 3 / cute_vectorized_timing / 1e9
        )
        cute_tv_layout_bandwidth = (
            shape * shape * dtype.itemsize * 3 / cute_tv_layout_timing / 1e9
        )
        torch_bandwidth = shape * shape * dtype.itemsize * 3 / torch_timing / 1e9

        df.loc[len(df)] = [
            shape,
            cute_timing,
            cute_vectorized_timing,
            cute_tv_layout_timing,
            torch_timing,
            cute_bandwidth,
            cute_vectorized_bandwidth,
            cute_tv_layout_bandwidth,
            torch_bandwidth,
        ]

    print(df)
    # df.to_csv("add_bench.csv", mode="a", header=False)
    # print(f"Shape: {shape}, CUTE Timing: {cute_timing:.3f} ms, CUTE Vectorized Timing: {cute_vectorized_timing:.3f} ms, Torch Timing: {torch_timing:.3f} ms, CUTE Bandwidth: {cute_bandwidth:.2f} GiB/s, CUTE Vectorized Bandwidth: {cute_vectorized_bandwidth:.2f} GiB/s, Torch Bandwidth: {torch_bandwidth:.2f} TB/s")


if __name__ == "__main__":
    bench_add()
