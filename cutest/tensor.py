from typing import Self
import torch

from cutest.ops.op import Op


class Item:
    def __init__(self, value: float | int):
        self.value = value
        self.op = None


class Tensor:
    def __init__(
        self,
        value: torch.Tensor
        | list[torch.Tensor]
        | tuple[Self, Item]
        | tuple[Item, Self],
        op: Op | None = None,
    ):
        self.value = value
        self.op = op

    def __add__(self, other: Self | Item) -> Self:
        from cutest.ops.elem_wise import Add

        op = Add()
        return Tensor([self, other], op)
