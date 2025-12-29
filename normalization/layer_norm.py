# No affine LayerNorm

import cutlass.cute as cute
import cutlass
import torch
from cutlass.cute.runtime import make_fake_compact_tensor


_kernel_cache: dict[tuple, object] = {}


def _get_config(n: int) -> tuple[int, int]:
    """Returns (threads_per_row, values_per_thread) for given N."""

    def is_valid(vpt, min_alignment=1):
        return vpt <= 64 and vpt != 16 and vpt % min_alignment == 0 # vpt 16 is giving segfaults in kernel

    for alignment in [8, 4, 1]:
        for tpr in [256, 128, 64, 32]:
            if n % tpr == 0:
                vpt = n // tpr
                if is_valid(vpt, alignment):
                    return (tpr, vpt)

    raise ValueError(f"N={n} not supported")


def _make_kernel(threads_per_row: int, values_per_thread: int, n_dim: int):

    num_warps = threads_per_row // 32

    @cute.kernel
    def layer_norm_device(
        gX: cute.Tensor,
        gY: cute.Tensor,
        epsilon: float,
        tv_layout,
    ):
        smem = cutlass.utils.SmemAllocator()
        smem_buf = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((num_warps,)),
        )

        t_idx, _, _ = cute.arch.thread_idx()
        b_idx, _, _ = cute.arch.block_idx()

        warp_id = t_idx // 32
        lane_id = t_idx % 32

        # partition input/output
        block_X = gX[((None, None), b_idx)]
        block_Y = gY[((None, None), b_idx)]

        tidfrgX = cute.composition(block_X, tv_layout)
        tidfrgY = cute.composition(block_Y, tv_layout)

        thrX = tidfrgX[(t_idx, None)]
        thrY = tidfrgY[(t_idx, None)]

        # load data
        regX = thrX.load()
        regX_f32 = regX.to(cutlass.Float32)

        # mean
        local_sum = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(values_per_thread):
            local_sum += regX_f32[i]

        warp_sum = cute.arch.warp_reduction(local_sum, op=lambda x, y: x + y, threads_in_group=32)

        if lane_id == 0:
            smem_buf[warp_id] = warp_sum
        cute.arch.sync_threads()

        if t_idx < num_warps:
            warp_sum = smem_buf[t_idx]
        else:
            warp_sum = cutlass.Float32(0.0)

        if t_idx < 32:
            total = cute.arch.warp_reduction(warp_sum, op=lambda x, y: x + y, threads_in_group=num_warps)
            if t_idx == 0:
                smem_buf[0] = total

        cute.arch.sync_threads()
        mean = smem_buf[0] / float(n_dim)

        # variance
        local_var = cutlass.Float32(0.0)
        for j in cutlass.range_constexpr(values_per_thread):
            diff = regX_f32[j] - mean
            local_var += diff * diff

        warp_var = cute.arch.warp_reduction(local_var, op=lambda x, y: x + y, threads_in_group=32)

        if lane_id == 0:
            smem_buf[warp_id] = warp_var
        cute.arch.sync_threads()

        if t_idx < num_warps:
            warp_var = smem_buf[t_idx]
        else:
            warp_var = cutlass.Float32(0.0)

        if t_idx < 32:
            total_var = cute.arch.warp_reduction(warp_var, op=lambda x, y: x + y, threads_in_group=num_warps)
            if t_idx == 0:
                smem_buf[0] = total_var

        cute.arch.sync_threads()
        variance = smem_buf[0] / float(n_dim)
        rstd = cute.math.rsqrt(variance + epsilon, fastmath=True)

        # normalize and store
        thrY[None] = ((regX_f32 - mean) * rstd).to(gY.element_type)

    @cute.jit
    def layer_norm_jit(
        x: cute.Tensor,
        y: cute.Tensor,
        epsilon: cutlass.Constexpr,
    ):
        t_layout = cute.make_ordered_layout((1, threads_per_row), order=(1, 0))
        v_layout = cute.make_ordered_layout((1, values_per_thread), order=(1, 0))
        tiler_mn, tv_layout = cute.make_layout_tv(t_layout, v_layout)

        tiled_X = cute.zipped_divide(x, tiler_mn)
        tiled_Y = cute.zipped_divide(y, tiler_mn)

        layer_norm_device(tiled_X, tiled_Y, epsilon, tv_layout).launch(
            grid=[cute.size(tiled_X, mode=[1]), 1, 1],
            block=[threads_per_row, 1, 1],
        )
        return y

    return layer_norm_jit


def _get_compiled_kernel(dtype: torch.dtype, n: int, eps: float):
    key = (dtype, n, eps)
    if key in _kernel_cache:
        return _kernel_cache[key]

    tpr, vpt = _get_config(n)
    m = cute.sym_int()

    cutlass_dtype = {
        torch.bfloat16: cutlass.BFloat16,
        torch.float16: cutlass.Float16,
        torch.float32: cutlass.Float32,
    }[dtype]

    fake_x = make_fake_compact_tensor(cutlass_dtype, (m, n), stride_order=(1, 0), assumed_align=16)
    fake_y = make_fake_compact_tensor(cutlass_dtype, (m, n), stride_order=(1, 0), assumed_align=16)

    jit_kernel = _make_kernel(tpr, vpt, n)
    fn = cute.compile(jit_kernel, fake_x, fake_y, float(eps), options="--enable-tvm-ffi")

    _kernel_cache[key] = fn
    return fn


def layer_norm_fwd(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y = torch.empty_like(x)
    fn = _get_compiled_kernel(x.dtype, x.shape[-1], eps)
    fn(x, y)
    return y
