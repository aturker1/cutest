from abc import ABC
import torch # Do we want torch?
from cutest.ops.op import Op


class Block(Op):
    def __init__(self, name: str):
        super().__init__(name)


class RMSNorm(Block):
    def __init__(self):
        super().__init__("rms_norm")

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
        raise NotImplementedError