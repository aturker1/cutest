import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
import cutlass


torch.set_printoptions(precision=4, sci_mode=False)


@cute.kernel
def rms_norm(
    gX: cute.Tensor, gY: cute.Tensor, gScale: cute.Tensor, epsilon: float, tv_layout
):
    t_idx, _, _ = cute.arch.thread_idx()
    b_idx, _, _ = cute.arch.block_idx()

    allocator = cutlass.utils.SmemAllocator()
    scalar_layout = cute.make_layout((1))
    squared_reduce = allocator.allocate_tensor(cutlass.Float32, layout=scalar_layout)

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

    mean_sq = cute.arch.warp_reduction(mean_sq, op=lambda x, y: x + y)
    if t_idx == 0:
        squared_reduce[0] = cute.math.rsqrt(mean_sq / 256.0 + epsilon, fastmath=True)

    cute.arch.sync_threads()

    thrY[None] = (regA * squared_reduce[0]).to(gY.element_type) * scale_val


@cute.jit
def rms_norm_kernel(
    x: cute.Tensor,
    y: cute.Tensor,
    scale: cute.Tensor,
    epsilon: cutlass.Constexpr,
):
    # vec_byte = 16
    # vec_len = (vec_byte * 8) // x.element_type.width

    t_layout = cute.make_ordered_layout((1, 32), order=(1, 0))  # 32 threads
    v_layout = cute.make_ordered_layout((1, 8), order=(1, 0))  # 8 elements each

    tiler_mn, tv_layout = cute.make_layout_tv(t_layout, v_layout)

    tiled_X = cute.zipped_divide(x, tiler_mn)
    tiled_Y = cute.zipped_divide(y, tiler_mn)
    tiled_scale = cute.zipped_divide(scale, tiler_mn)

    rms_norm(tiled_X, tiled_Y, tiled_scale, epsilon, tv_layout).launch(
        grid=[cute.size(tiled_X, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )

    return y


if __name__ == "__main__":
    x = torch.randn((1024, 256), dtype=torch.bfloat16, device="cuda")
    y = torch.zeros((1024, 256), dtype=torch.bfloat16, device="cuda")
    scale = torch.randn((256,), dtype=torch.bfloat16, device="cuda")

    x_cute = from_dlpack(x, assumed_align=16)
    y_cute = from_dlpack(y, assumed_align=16)
    scale_cute = from_dlpack(scale.view((1, scale.shape[0])), assumed_align=16)
    epsilon = 1e-6

    kernel = cute.compile(rms_norm_kernel, x_cute, y_cute, scale_cute, epsilon)

    ref_res = torch.nn.functional.rms_norm(x, (256,), weight=scale, eps=epsilon)
    kernel_res = kernel(x_cute, y_cute, scale_cute)

    # Benchmark the kernel
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(100):
        kernel_res = kernel(x_cute, y_cute, scale_cute)
    end_event.record()
    torch.cuda.synchronize()

    print(f"Kernel execution time: {start_event.elapsed_time(end_event)} ms")

    # Benchmark the reference
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(100):
        ref_res = torch.nn.functional.rms_norm(x, (256,), weight=scale, eps=epsilon)
    end_event.record()
    torch.cuda.synchronize()
    print(f"Reference execution time: {start_event.elapsed_time(end_event)} ms")

    # MAE

    print(f"MAE: {torch.mean(torch.abs(ref_res - y))}")
