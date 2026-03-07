from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from cutest.tensor import Item, Tensor

try:
    from graphviz import Digraph
except ImportError:  # pragma: no cover
    Digraph = None

GraphValue = Tensor | Item
_SKIP_OP_ATTRS = {"name", "output", "outputs", "result", "results"}


class GraphVisualization:
    def __init__(self, graph: Digraph):
        self.graph = graph

    def _repr_svg_(self) -> str:
        return self.to_svg()

    def __str__(self) -> str:
        return self.graph.source

    def to_svg(self) -> str:
        return self.graph.pipe(format="svg").decode("utf-8")

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.write_text(self.to_svg(), encoding="utf-8")
        return output


def visualize_graph(values: GraphValue | Iterable[GraphValue]) -> GraphVisualization:
    if Digraph is None:
        raise ImportError("visualize_graph() requires `graphviz` to be installed")

    graph = Digraph("cutest")
    graph.attr(rankdir="LR")
    graph.attr("node", fontname="Helvetica")
    _build_graph(graph, _normalize_values(values))
    return GraphVisualization(graph)


def _normalize_values(values: GraphValue | Iterable[GraphValue]) -> list[GraphValue]:
    if isinstance(values, (Tensor, Item)):
        return [values]

    values = list(values)
    if not values:
        raise ValueError("visualize_graph() expected at least one Tensor or Item")
    if not all(isinstance(value, (Tensor, Item)) for value in values):
        raise TypeError(
            "visualize_graph() only accepts Tensor, Item, or iterables of them"
        )
    return values


def _build_graph(graph: Digraph, roots: list[GraphValue]) -> None:
    seen_values: set[int] = set()
    seen_ops: set[int] = set()
    seen_edges: set[tuple[str, str]] = set()

    def add_edge(src: str, dst: str) -> None:
        edge = (src, dst)
        if edge not in seen_edges:
            seen_edges.add(edge)
            graph.edge(src, dst)

    def visit_value(value: GraphValue) -> str:
        node_id = f"value_{id(value)}"
        if id(value) in seen_values:
            return node_id

        seen_values.add(id(value))
        kind = "tensor" if isinstance(value, Tensor) else "item"
        graph.node(node_id, _value_label(value), **_node_style(kind))

        if value.op is not None:
            op_id = visit_op(value.op)
            for input_value in _value_inputs(value):
                add_edge(visit_value(input_value), op_id)
            add_edge(op_id, node_id)

        return node_id

    def visit_op(op: object) -> str:
        node_id = f"op_{id(op)}"
        if id(op) in seen_ops:
            return node_id

        seen_ops.add(id(op))
        graph.node(
            node_id,
            f"Op\n{getattr(op, 'name', op.__class__.__name__)}",
            **_node_style("op"),
        )
        return node_id

    for root in roots:
        visit_value(root)


def _op_inputs(op: object) -> list[GraphValue]:
    inputs: list[GraphValue] = []
    seen: set[int] = set()
    for name, value in vars(op).items():
        if name.startswith("_") or name in _SKIP_OP_ATTRS:
            continue
        for nested in _iter_values(value):
            if id(nested) not in seen:
                seen.add(id(nested))
                inputs.append(nested)
    return inputs


def _tensor_inputs(tensor: Tensor) -> list[GraphValue]:
    return list(_iter_values(tensor.value))


def _value_inputs(value: GraphValue) -> list[GraphValue]:
    if isinstance(value, Item) or value.op is None:
        return []

    inputs: list[GraphValue] = []
    seen: set[int] = set()
    for source in (_tensor_inputs(value), _op_inputs(value.op)):
        for input_value in source:
            if id(input_value) not in seen:
                seen.add(id(input_value))
                inputs.append(input_value)
    return inputs


def _iter_values(value: object):
    if isinstance(value, (Tensor, Item)):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _iter_values(nested)
    elif isinstance(value, (list, tuple, set)):
        for nested in value:
            yield from _iter_values(nested)


def _value_label(value: GraphValue) -> str:
    if isinstance(value, Tensor):
        tensor = value.value
        if isinstance(tensor, torch.Tensor):
            shape = "scalar" if tensor.ndim == 0 else "x".join(map(str, tensor.shape))
            return f"Tensor\nshape={shape}\ndtype={tensor.dtype}"
        if _tensor_inputs(value):
            return "Tensor"
        return f"Tensor\n{type(tensor).__name__}"
    return f"Item\nvalue={value.value}"


def _node_style(kind: str) -> dict[str, str]:
    return {
        "tensor": {
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "#dbeafe",
            "color": "#2563eb",
        },
        "item": {
            "shape": "ellipse",
            "style": "filled",
            "fillcolor": "#fef3c7",
            "color": "#d97706",
        },
        "op": {
            "shape": "diamond",
            "style": "filled",
            "fillcolor": "#dcfce7",
            "color": "#16a34a",
        },
    }[kind]
