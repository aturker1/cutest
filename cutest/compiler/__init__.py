from cutest.compiler.trace import trace, FusedIR, IRNode
from cutest.compiler.codegen import codegen
from cutest.compiler.execute import execute

__all__ = ["trace", "codegen", "fuse", "execute", "FusedIR", "IRNode"]


def fuse(tensor):
    """Trace a computation graph and generate a fused CuTe DSL kernel string."""
    ir = trace(tensor)
    return codegen(ir)
