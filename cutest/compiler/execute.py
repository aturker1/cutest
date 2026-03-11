from __future__ import annotations

import importlib
import os
import sys
import tempfile

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor

from cutest.compiler.codegen import codegen
from cutest.compiler.trace import FusedIR, trace

ASSUMED_ALIGN_BYTES = 16

# Cache compiled kernels by (dtype, shape, op_signature)
_kernel_cache: dict[tuple, object] = {}

# Persistent temp dir for generated kernel files
_kernel_dir = tempfile.mkdtemp(prefix="cutest_kernels_")


def _torch_dtype_to_cutlass(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.float32:
        return cutlass.Float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _op_signature(ir: FusedIR) -> tuple:
    """Hashable signature of the op graph (for caching)."""
    return tuple((n.name, n.op, tuple(n.inputs)) for n in ir.nodes if n.op is not None)


def _load_kernel_module(code: str, module_name: str):
    """Write generated code to a file and import it so CuTe can inspect source."""
    filepath = os.path.join(_kernel_dir, f"{module_name}.py")
    with open(filepath, "w") as f:
        f.write(code)

    spec = importlib.util.spec_from_file_location(module_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _compile_ir(ir: FusedIR) -> object:
    """Compile the IR into a callable TVM-FFI function."""
    first_tensor = ir.input_tensors[ir.inputs[0]]
    dtype = first_tensor.dtype
    m, n = first_tensor.shape[0], first_tensor.shape[1]

    sig = _op_signature(ir)
    cache_key = (_torch_dtype_to_cutlass(dtype), int(m), int(n), sig)
    cached = _kernel_cache.get(cache_key)
    if cached is not None:
        return cached

    # Generate code, write to file, import (CuTe needs inspect.getsource)
    code = codegen(ir)
    module_name = f"_cutest_fused_{abs(hash(sig)) % 10**8}"
    mod = _load_kernel_module(code, module_name)
    jit_fn = mod.fused_elementwise_kernel

    # Build fake tensors for TVM-FFI compilation
    cutlass_dtype = _torch_dtype_to_cutlass(dtype)
    fake_tensors = []
    for _ in ir.inputs:
        fake_tensors.append(
            make_fake_compact_tensor(
                cutlass_dtype,
                (int(m), int(n)),
                stride_order=(1, 0),
                assumed_align=ASSUMED_ALIGN_BYTES,
            )
        )
    fake_tensors.append(
        make_fake_compact_tensor(
            cutlass_dtype,
            (int(m), int(n)),
            stride_order=(1, 0),
            assumed_align=ASSUMED_ALIGN_BYTES,
        )
    )

    fn = cute.compile(jit_fn, *fake_tensors, options="--enable-tvm-ffi")
    _kernel_cache[cache_key] = fn
    return fn


def compile(tensor):
    """Trace and compile once. Returns a callable: f(*input_tensors) -> output_tensor."""
    ir = trace(tensor)
    fn = _compile_ir(ir)
    n_inputs = len(ir.inputs)

    def _run(*inputs: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(inputs[0])
        fn(*inputs, out)
        return out

    return _run


def execute(tensor) -> torch.Tensor:
    """Trace, compile, and run a fused elementwise kernel. Returns output tensor."""
    ir = trace(tensor)
    fn = _compile_ir(ir)

    first_tensor = ir.input_tensors[ir.inputs[0]]
    out = torch.empty_like(first_tensor)

    args = [ir.input_tensors[name] for name in ir.inputs] + [out]
    fn(*args)
    return out
