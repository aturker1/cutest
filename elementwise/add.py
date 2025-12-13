import cutlass.cute as cute


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
