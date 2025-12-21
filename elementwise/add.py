import cutlass.cute as cute
import cutlass
import torch
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor


ASSUMED_ALIGN_BYTES = 16

# Cache TVM-FFI compiled kernels by (dtype, m, n)
_tvm_ffi_kernel_cache: dict[tuple, object] = {}
vectorized_kernel_cache: dict[tuple, object] = {}


def _torch_dtype_to_cutlass(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.float32:
        return cutlass.Float32
    raise ValueError(f"Unsupported dtype: {dtype}")


@cute.kernel
def naive_add(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    t_idx, _, _ = cute.arch.thread_idx()
    b_idx, _, _ = cute.arch.block_idx()
    b_dim, _, _ = cute.arch.block_dim()

    m, n = gA.shape
    idx = t_idx + b_idx * b_dim

    if idx < m * n:
        m_idx = idx // n
        n_idx = idx % n
        gC[m_idx, n_idx] = gA[m_idx, n_idx] + gB[m_idx, n_idx]


@cute.jit
def naive_add_kernel(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
):
    n_threads_per_block = 256

    m = A.shape[0]
    n = A.shape[1]

    kernel = naive_add(A, B, C)

    kernel.launch(
        grid=((m * n) // n_threads_per_block, 1, 1),
        block=(n_threads_per_block, 1, 1),
    )

    return C


@cute.kernel
def vectorized_add(
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
def vectorized_add_kernel(
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

    kernel = vectorized_add(gA, gB, gC)

    kernel.launch(
        grid=((m * n) // n_threads_per_block, 1, 1),  # Number of blocks in x,y,z
        block=(n_threads_per_block, 1, 1),  # Threads per block in x,y,z
    )

    return C


def vectorized_add_wrapper(A: torch.Tensor, B: torch.Tensor):
    A_cute = from_dlpack(A, assumed_align=ASSUMED_ALIGN_BYTES)
    B_cute = from_dlpack(B, assumed_align=ASSUMED_ALIGN_BYTES)
    C = torch.empty_like(A)
    C_cute = from_dlpack(C, assumed_align=ASSUMED_ALIGN_BYTES)
    key = (A.dtype, A.shape[0], A.shape[1])
    if key not in vectorized_kernel_cache:
        vectorized_kernel_cache[key] = cute.compile(
            vectorized_add_kernel, A_cute, B_cute, C_cute
        )
    vectorized_kernel_cache[key](A_cute, B_cute, C_cute)
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
        vectorized_add_kernel, fake_A, fake_B, fake_C, options="--enable-tvm-ffi"
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


@cute.kernel
def tv_layout_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    tv_layout: cute.Layout,
):
    t_idx, _, _ = cute.arch.thread_idx()
    b_idx, _, _ = cute.arch.block_idx()

    block_coord = ((None, None), b_idx)

    block_A = gA[block_coord]
    block_B = gB[block_coord]
    block_C = gC[block_coord]

    tidfrgA = cute.composition(block_A, tv_layout)
    tidfrgB = cute.composition(block_B, tv_layout)
    tidfrgC = cute.composition(block_C, tv_layout)

    thr_coord = (t_idx, None)

    thrA = tidfrgA[thr_coord]
    thrB = tidfrgB[thr_coord]
    thrC = tidfrgC[thr_coord]

    thrC[None] = thrA.load() + thrB.load()


@cute.jit
def tv_layout_add(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
):
    # Block size 64,512

    vec_byte = 16
    vec_len = (
        vec_byte * 8
    ) // A.element_type.width  # width is in bits, convert properly

    t_layout = cute.make_ordered_layout((32 // 16, 1024 // vec_len), order=(1, 0))
    v_layout = cute.make_ordered_layout((16, vec_len), order=(1, 0))

    tiler_mn, tv_layout = cute.make_layout_tv(t_layout, v_layout)

    tiled_A = cute.zipped_divide(A, tiler_mn)
    tiled_B = cute.zipped_divide(B, tiler_mn)
    tiled_C = cute.zipped_divide(C, tiler_mn)

    tv_layout_add_kernel(tiled_A, tiled_B, tiled_C, tv_layout).launch(
        grid=[cute.size(tiled_C, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )

    return C
