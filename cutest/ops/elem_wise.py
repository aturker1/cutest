from cutest.tensor import Tensor, Item
from cutest.ops.op import Op


class ElementWise(Op):
    def __init__(self, op: str):
        super().__init__(op)

    def forward(self, a: Tensor | Item, b: Tensor | Item):
        return Tensor([a, b], self.op)


class Add(ElementWise):
    def __init__(self):
        super().__init__("+")


class Sub(ElementWise):
    def __init__(self):
        super().__init__("-")


class Mul(ElementWise):
    def __init__(self):
        super().__init__("*")


class Div(ElementWise):
    def __init__(self):
        super().__init__("/")
