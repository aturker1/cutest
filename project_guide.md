# The aim of the project
## Phase 1
We are aimimg here a simple compiler like engine that builds an graph via building blocks(add, substractract etc.) fuse them into one kernel and write down the cutedsl kernel. 

```python

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

```

As you can see above this is SOTA C=A+B elementwise cutedsl implementation. First step would be make it support 2 element wise operation automatically with the cutest/ops/elem_wise.py operations here. Wrote ex.py but this is too complicated for phase 1, we just need to think elementwise operations.