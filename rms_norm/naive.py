import cutlass.cute as cute
import torch
import cutlass
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor


torch.backends.cuda.matmul.allow_tf32 = True


ASSUMED_ALIGN_BYTES = 16

_rms_norm_kernel_cache: dict[tuple, object] = {}
_tvm_ffi_kernel_cache: dict[tuple, object] = {}


def _torch_dtype_to_cutlass(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.float32:
        return cutlass.Float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _get_rms_norm_kernel_tvm_ffi(*, dtype: torch.dtype, n: int, eps: float):
    key = (dtype, int(n), float(eps), ASSUMED_ALIGN_BYTES)

    assert int(n) == 128, "Only supported for n=128"

    fn = _tvm_ffi_kernel_cache.get(key)
    if fn is not None:
        return fn

    m = cute.sym_int()
    cutlass_dtype = _torch_dtype_to_cutlass(dtype)
    fake_x = make_fake_compact_tensor(
        cutlass_dtype,
        (m, int(n)),
        stride_order=(1, 0),  # row-major (matches torch contiguous)
        assumed_align=ASSUMED_ALIGN_BYTES,
    )
    fake_y = make_fake_compact_tensor(
        cutlass_dtype,
        (m, int(n)),
        stride_order=(1, 0),
        assumed_align=ASSUMED_ALIGN_BYTES,
    )
    fake_w = make_fake_compact_tensor(
        cutlass_dtype,
        (int(n),),
        stride_order=(0,),
        assumed_align=ASSUMED_ALIGN_BYTES,
    )

    fn = cute.compile(
        rms_norm_kernel, fake_x, fake_y, fake_w, float(eps), options="--enable-tvm-ffi"
    )
    _tvm_ffi_kernel_cache[key] = fn
    return fn


@cute.kernel
def rms_norm(
    gX: cute.Tensor,
    gY: cute.Tensor,
    gScale: cute.Tensor,
    epsilon: float,
    tv_layout,
    tv_scale,
):
    t_idx, _, _ = cute.arch.thread_idx()
    b_idx, _, _ = cute.arch.block_idx()

    thr_coord = (t_idx, None)
    block_coord = ((None, None), b_idx)

    block_X = gX[block_coord]
    block_Y = gY[block_coord]

    tidfrgX = cute.composition(block_X, tv_layout)
    tidfrgY = cute.composition(block_Y, tv_layout)
    tidfrgScale = cute.composition(gScale, tv_scale)

    thrX = tidfrgX[thr_coord]
    thrY = tidfrgY[thr_coord]

    scale_val = tidfrgScale[(t_idx % 16, None)].load()

    regA = thrX.load()

    mean_sq = 0.0

    for i in range(regA.shape[0]):
        mean_sq += regA[i] * regA[i]

    mean_sq = cute.arch.warp_reduction(
        mean_sq, op=lambda x, y: x + y, threads_in_group=16
    )

    reduced_sqrt = cute.math.rsqrt(mean_sq / 128.0 + epsilon, fastmath=True)

    thrY[None] = (regA * reduced_sqrt * scale_val.to(cutlass.Float32)).to(
        gY.element_type
    )


@cute.jit
def rms_norm_kernel(
    x: cute.Tensor,
    y: cute.Tensor,
    scale: cute.Tensor,
    epsilon: cutlass.Constexpr,
):
    t_layout = cute.make_ordered_layout((8, 16), order=(1, 0))
    v_layout = cute.make_ordered_layout((1, 8), order=(1, 0))

    tiler_mn, tv_layout = cute.make_layout_tv(t_layout, v_layout)
    tiler_scale, tv_scale = cute.make_layout_tv(
        cute.make_ordered_layout((16,), order=(0,)),
        cute.make_ordered_layout((8,), order=(0,)),
    )

    tiled_X = cute.zipped_divide(x, tiler_mn)
    tiled_Y = cute.zipped_divide(y, tiler_mn)
    tiled_scale = cute.zipped_divide(scale, tiler_scale)

    rms_norm(tiled_X, tiled_Y, tiled_scale, epsilon, tv_layout, tv_scale).launch(
        grid=[cute.size(tiled_X, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )

    return y


def rms_norm_fwd(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    y = torch.empty_like(x)

    x_cute = from_dlpack(x, assumed_align=ASSUMED_ALIGN_BYTES)
    w_cute = from_dlpack(w, assumed_align=ASSUMED_ALIGN_BYTES)
    y_cute = from_dlpack(y, assumed_align=ASSUMED_ALIGN_BYTES)

    key = (x.shape, w.shape, eps)
    if key in _rms_norm_kernel_cache:
        _rms_norm_kernel_cache[key](x_cute, y_cute, w_cute)
        return y

    kernel = cute.compile(rms_norm_kernel, x_cute, y_cute, w_cute, eps)
    _rms_norm_kernel_cache[key] = kernel
    kernel(x_cute, y_cute, w_cute)
    return y


def rms_norm_fwd_tvm_ffi(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    y = torch.empty_like(x)

    fn = _get_rms_norm_kernel_tvm_ffi(
        dtype=x.dtype, m=int(x.shape[0]), n=int(x.shape[1]), eps=float(eps)
    )
    fn(x, y, w)
    return y


def _get_rms_norm_kernel_tvm_ffi(*, dtype: torch.dtype, m: int, n: int, eps: float):
    """TVM-FFI kernel with CONCRETE M dimension (not symbolic)."""
    key = (dtype, int(m), int(n), float(eps), ASSUMED_ALIGN_BYTES)

    fn = _tvm_ffi_kernel_cache.get(key)
    if fn is not None:
        return fn

    cutlass_dtype = _torch_dtype_to_cutlass(dtype)
    fake_x = make_fake_compact_tensor(
        cutlass_dtype,
        (int(m), int(n)),  # CONCRETE m, not symbolic
        stride_order=(1, 0),
        assumed_align=ASSUMED_ALIGN_BYTES,
    )
    fake_y = make_fake_compact_tensor(
        cutlass_dtype,
        (int(m), int(n)),
        stride_order=(1, 0),
        assumed_align=ASSUMED_ALIGN_BYTES,
    )
    fake_w = make_fake_compact_tensor(
        cutlass_dtype,
        (int(n),),
        stride_order=(0,),
        assumed_align=ASSUMED_ALIGN_BYTES,
    )

    fn = cute.compile(
        rms_norm_kernel, fake_x, fake_y, fake_w, float(eps), options="--enable-tvm-ffi"
    )
    _tvm_ffi_kernel_cache[key] = fn
    return fn
