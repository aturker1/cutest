import torch

from cutest.ops.op import Op

class Tensor:
    def __init__(self, value: torch.Tensor, op: Op | None = None):
        self.value = value
        self.op = op

class Item:
    def __init__(self, value: float | int):
        self.value = value
        self.op = None