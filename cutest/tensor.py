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

        return Tensor([self, other], Add())

    def __sub__(self, other: Self | Item) -> Self:
        from cutest.ops.elem_wise import Sub

        return Tensor([self, other], Sub())

    def __mul__(self, other: Self | Item) -> Self:
        from cutest.ops.elem_wise import Mul

        return Tensor([self, other], Mul())

    def __truediv__(self, other: Self | Item) -> Self:
        from cutest.ops.elem_wise import Div

        return Tensor([self, other], Div())
