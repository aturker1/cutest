import cutlass.cute as cute


@cute.kernel
def naive_add(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    # 1D kernel
    t_idx, _, _ = cute.arch.thread_idx()
    b_idx, _, _ = cute.arch.block_idx()
    b_dim, _, _ = cute.arch.block_dim()

    m, n = gA.shape
    idx = t_idx + b_idx * b_dim

    # Bounds check to prevent illegal memory access
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
    n_threads_per_block = 128

    m = A.shape[0]
    n = A.shape[1]

    kernel = naive_add(A, B, C)

    kernel.launch(
        grid=((m * n) // n_threads_per_block, 1, 1),  # Number of blocks in x,y,z
        block=(n_threads_per_block, 1, 1),  # Threads per block in x,y,z
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

    # Map thread index to logical index of input tensor in unit of vector
    m, n = gA.shape[1]  # thread-domain
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
    n_threads_per_block = 128

    gA = cute.zipped_divide(A, (1, 4))
    gB = cute.zipped_divide(B, (1, 4))
    gC = cute.zipped_divide(C, (1, 4))

    m = gA.shape[1][0]
    n = gA.shape[1][1]

    kernel = vectorized_add(gA, gB, gC)

    kernel.launch(
        grid=((m * n) // n_threads_per_block, 1, 1),  # Number of blocks in x,y,z
        block=(n_threads_per_block, 1, 1),  # Threads per block in x,y,z
    )

    return C
