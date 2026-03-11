from __future__ import annotations

from cutest.compiler.trace import FusedIR


def codegen(ir: FusedIR) -> str:
    """Generate a fused CuTe DSL kernel from elementwise IR."""
    input_params = [f"g{name}: cute.Tensor" for name in ir.inputs]
    all_params = input_params + ["gOut: cute.Tensor"]

    # --- @cute.kernel (per-thread body) ---
    kernel_lines = []
    kernel_lines.append("@cute.kernel")
    kernel_lines.append(f"def _fused_elementwise(")
    for p in all_params:
        kernel_lines.append(f"    {p},")
    kernel_lines.append("):")

    # Thread indexing
    kernel_lines.append("    t_idx, _, _ = cute.arch.thread_idx()")
    kernel_lines.append("    b_idx, _, _ = cute.arch.block_idx()")
    kernel_lines.append("    b_dim, _, _ = cute.arch.block_dim()")
    kernel_lines.append("")
    kernel_lines.append("    idx = t_idx + b_idx * b_dim")
    kernel_lines.append(f"    m, n = g{ir.inputs[0]}.shape[1]")
    kernel_lines.append("    ni = idx % n")
    kernel_lines.append("    mi = idx // n")
    kernel_lines.append("")

    # Load inputs
    for name in ir.inputs:
        kernel_lines.append(
            f"    {name} = g{name}[(None, (mi, ni))].load()"
        )
    kernel_lines.append("")

    # Compute ops (skip input-only nodes)
    for node in ir.nodes:
        if node.op is not None:
            lhs, rhs = node.inputs
            kernel_lines.append(f"    {node.name} = {lhs} {node.op} {rhs}")

    kernel_lines.append("")

    # Store output
    kernel_lines.append(f"    gOut[(None, (mi, ni))] = {ir.output}")

    # --- @cute.jit (launch wrapper) ---
    jit_params = [f"{name}: cute.Tensor" for name in ir.inputs]
    jit_all_params = jit_params + ["Out: cute.Tensor"]

    jit_lines = []
    jit_lines.append("")
    jit_lines.append("")
    jit_lines.append("@cute.jit")
    jit_lines.append("def fused_elementwise_kernel(")
    for p in jit_all_params:
        jit_lines.append(f"    {p},")
    jit_lines.append("):")
    jit_lines.append("    n_threads_per_block = 256")
    jit_lines.append("")

    # Zipped divide for all tensors
    for name in ir.inputs:
        jit_lines.append(f"    g{name} = cute.zipped_divide({name}, (4, 8))")
    jit_lines.append("    gOut = cute.zipped_divide(Out, (4, 8))")
    jit_lines.append("")

    jit_lines.append(f"    m = g{ir.inputs[0]}.shape[1][0]")
    jit_lines.append(f"    n = g{ir.inputs[0]}.shape[1][1]")
    jit_lines.append("")

    # Launch call
    launch_args = ", ".join(f"g{name}" for name in ir.inputs) + ", gOut"
    jit_lines.append(f"    _fused_elementwise({launch_args}).launch(")
    jit_lines.append("        grid=((m * n) // n_threads_per_block, 1, 1),")
    jit_lines.append("        block=(n_threads_per_block, 1, 1),")
    jit_lines.append("    )")
    jit_lines.append("")
    jit_lines.append("    return Out")

    # Combine
    header = "import cutlass.cute as cute\n\n"
    return header + "\n".join(kernel_lines) + "\n".join(jit_lines) + "\n"
