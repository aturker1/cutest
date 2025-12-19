import cutlass.cute as cute
import torch
import cutlass


torch.set_printoptions(precision=4, sci_mode=False)
# torch enable fastmath
torch.backends.cuda.matmul.allow_tf32 = True


@cute.kernel
def rms_norm(
    gX: cute.Tensor, gY: cute.Tensor, gScale: cute.Tensor, epsilon: float, tv_layout
):
    t_idx, _, _ = cute.arch.thread_idx()
    b_idx, _, _ = cute.arch.block_idx()

    block_coord = ((None, None), b_idx)

    block_X = gX[block_coord]
    block_Y = gY[block_coord]

    tidfrgX = cute.composition(block_X, tv_layout)
    tidfrgY = cute.composition(block_Y, tv_layout)

    thr_coord = (t_idx, None)
    thrX = tidfrgX[thr_coord]
    thrY = tidfrgY[thr_coord]
    block_scale = gScale[((None, None), 0)]
    tidfrgScale = cute.composition(block_scale, tv_layout)
    thrScale = tidfrgScale[thr_coord]
    scale_val = thrScale.load()

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
    t_layout = cute.make_ordered_layout((16, 16), order=(1, 0))
    v_layout = cute.make_ordered_layout((1, 8), order=(1, 0))

    tiler_mn, tv_layout = cute.make_layout_tv(t_layout, v_layout)

    tiled_X = cute.zipped_divide(x, tiler_mn)
    tiled_Y = cute.zipped_divide(y, tiler_mn)
    tiled_scale = cute.zipped_divide(scale, tiler_mn)

    rms_norm(tiled_X, tiled_Y, tiled_scale, epsilon, tv_layout).launch(
        grid=[cute.size(tiled_X, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )

    return y
