from abc import ABC
from cutest.tensor import Tensor, Item
from cutest.ops.op import Op

class ElementWise(Op):
    def __init__(self, name: str):
        super().__init__(name)
        

class Add(ElementWise):
    def __init__(self):
        super().__init__("add")


    def forward(self, a: Tensor | Item, b: Tensor | Item):
        return 