from abc import ABC

class Op(ABC):
    def __init__(self, name: str):
        self.name = name