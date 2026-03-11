from __future__ import annotations

from dataclasses import dataclass, field

import torch

from cutest.ops.elem_wise import ElementWise


@dataclass
class IRNode:
    name: str
    op: str | None = None  # "+", "-", "*", "/" or None for inputs
    inputs: list[str] = field(default_factory=list)


@dataclass
class FusedIR:
    inputs: list[str]  # ordered input names: ["in0", "in1", ...]
    nodes: list[IRNode]  # topologically sorted (inputs first, then ops)
    output: str  # name of the final output node
    input_tensors: dict[str, torch.Tensor] = field(
        default_factory=dict
    )  # name -> actual tensor


def trace(tensor) -> FusedIR:
    """Walk a Tensor computation graph and extract a flat elementwise IR."""
    from cutest.tensor import Tensor

    visited: dict[int, str] = {}  # id(tensor) -> node name
    input_names: list[str] = []
    input_tensors: dict[str, torch.Tensor] = {}
    nodes: list[IRNode] = []
    _input_counter = [0]
    _op_counter = [0]

    def _walk(t) -> str:
        tid = id(t)
        if tid in visited:
            return visited[tid]

        if not isinstance(t, Tensor):
            raise TypeError(f"Expected Tensor, got {type(t)}")

        # Leaf tensor (concrete torch.Tensor, no op)
        if isinstance(t.value, torch.Tensor) and t.op is None:
            name = f"in{_input_counter[0]}"
            _input_counter[0] += 1
            visited[tid] = name
            input_names.append(name)
            input_tensors[name] = t.value
            nodes.append(IRNode(name=name))
            return name

        # Intermediate node produced by an elementwise op
        if isinstance(t.value, list) and isinstance(t.op, ElementWise):
            lhs_name = _walk(t.value[0])
            rhs_name = _walk(t.value[1])

            name = f"v{_op_counter[0]}"
            _op_counter[0] += 1
            visited[tid] = name
            nodes.append(IRNode(name=name, op=t.op.name, inputs=[lhs_name, rhs_name]))
            return name

        raise ValueError(
            f"Unsupported node: value type={type(t.value)}, op={t.op}. "
            "Phase 1 only supports elementwise ops on leaf tensors."
        )

    output_name = _walk(tensor)
    return FusedIR(
        inputs=input_names,
        nodes=nodes,
        output=output_name,
        input_tensors=input_tensors,
    )
