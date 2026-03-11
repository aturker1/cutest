import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor


ASSUMED_ALIGN_BYTES = 16

# Cache TVM-FFI compiled kernels by (dtype, m, n, alignment).
_tvm_ffi_kernel_cache: dict[tuple, object] = {}


def _torch_dtype_to_cutlass(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.float32:
        return cutlass.Float32
    raise ValueError(f"Unsupported dtype: {dtype}")


@cute.kernel
def _vectorized_add(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    t_idx, _, _ = cute.arch.thread_idx()
    b_idx, _, _ = cute.arch.block_idx()
    b_dim, _, _ = cute.arch.block_dim()

    idx = t_idx + b_idx * b_dim

    m, n = gA.shape[1]
    ni = idx % n
    mi = idx // n

    a_val = gA[(None, (mi, ni))].load()
    b_val = gB[(None, (mi, ni))].load()

    gC[(None, (mi, ni))] = a_val + b_val


@cute.jit
def _vectorized_add_kernel(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
):
    n_threads_per_block = 256

    gA = cute.zipped_divide(A, (4, 8))
    gB = cute.zipped_divide(B, (4, 8))
    gC = cute.zipped_divide(C, (4, 8))

    m = gA.shape[1][0]
    n = gA.shape[1][1]

    _vectorized_add(gA, gB, gC).launch(
        grid=((m * n) // n_threads_per_block, 1, 1),
        block=(n_threads_per_block, 1, 1),
    )

    return C


def _get_vectorized_add_kernel_tvm_ffi(*, dtype: torch.dtype, m: int, n: int):
    key = (dtype, int(m), int(n), ASSUMED_ALIGN_BYTES)
    fn = _tvm_ffi_kernel_cache.get(key)
    if fn is not None:
        return fn

    cutlass_dtype = _torch_dtype_to_cutlass(dtype)
    fake_A = make_fake_compact_tensor(
        cutlass_dtype,
        (int(m), int(n)),
        stride_order=(1, 0),
        assumed_align=ASSUMED_ALIGN_BYTES,
    )
    fake_B = make_fake_compact_tensor(
        cutlass_dtype,
        (int(m), int(n)),
        stride_order=(1, 0),
        assumed_align=ASSUMED_ALIGN_BYTES,
    )
    fake_C = make_fake_compact_tensor(
        cutlass_dtype,
        (int(m), int(n)),
        stride_order=(1, 0),
        assumed_align=ASSUMED_ALIGN_BYTES,
    )

    fn = cute.compile(
        _vectorized_add_kernel, fake_A, fake_B, fake_C, options="--enable-tvm-ffi"
    )
    _tvm_ffi_kernel_cache[key] = fn
    return fn


def vectorized_add_fwd_tvm_ffi(
    A: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    C = torch.empty_like(A)
    fn = _get_vectorized_add_kernel_tvm_ffi(
        dtype=A.dtype, m=int(A.shape[0]), n=int(A.shape[1])
    )
    fn(A, B, C)
    return C
