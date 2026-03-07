from cutest.ops.op import Op
from cutest.tensor import Tensor


class Block(Op):
    def __init__(self, name: str):
        super().__init__(name)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class RMSNorm(Block):
    def __init__(self, weight: Tensor, bias: Tensor, eps: float = 1e-5):
        super().__init__("rms_norm")
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def forward(self, x: Tensor):
        self.x = x
        return Tensor(x.value, self)
