"""Demo: Phase 1 kernel fusion - trace, codegen, compile, and execute."""

import torch
from cutest import Tensor, fuse, execute


def demo_three_ops():
    """E = (A + B) * C - D  -- generate kernel + run it"""
    a = torch.randn(256, 256, device="cuda", dtype=torch.float32)
    b = torch.randn(256, 256, device="cuda", dtype=torch.float32)
    c = torch.randn(256, 256, device="cuda", dtype=torch.float32)
    d = torch.randn(256, 256, device="cuda", dtype=torch.float32)

    # Build computation graph
    A, B, C, D = Tensor(a), Tensor(b), Tensor(c), Tensor(d)
    E = (A + B) * C - D

    # Show generated kernel
    print("=== Generated CuTe DSL Kernel: E = (A + B) * C - D ===")
    print(fuse(E))

    # Compile and execute
    result = execute(E)

    # Verify against PyTorch
    expected = (a + b) * c - d
    print(f"Max error: {(result - expected).abs().max().item():.2e}")
    assert torch.allclose(result, expected, atol=1e-5), "Mismatch!"
    print("PASSED - fused kernel matches PyTorch output\n")


def demo_simple_add():
    """C = A + B"""
    a = torch.randn(256, 256, device="cuda", dtype=torch.float32)
    b = torch.randn(256, 256, device="cuda", dtype=torch.float32)

    A, B = Tensor(a), Tensor(b)
    C = A + B

    result = execute(C)
    expected = a + b
    print("=== Simple Add: C = A + B ===")
    print(f"Max error: {(result - expected).abs().max().item():.2e}")
    assert torch.allclose(result, expected, atol=1e-5)
    print("PASSED\n")


def demo_all_ops():
    """F = (A + B) * C - D / A  -- all four ops"""
    a = torch.randn(256, 256, device="cuda", dtype=torch.float32) + 1  # avoid div by ~0
    b = torch.randn(256, 256, device="cuda", dtype=torch.float32)
    c = torch.randn(256, 256, device="cuda", dtype=torch.float32)
    d = torch.randn(256, 256, device="cuda", dtype=torch.float32)

    A, B, C, D = Tensor(a), Tensor(b), Tensor(c), Tensor(d)
    F = (A + B) * C - D / A

    result = execute(F)
    expected = (a + b) * c - d / a
    print("=== All ops: F = (A + B) * C - D / A ===")
    print(f"Max error: {(result - expected).abs().max().item():.2e}")
    assert torch.allclose(result, expected, atol=1e-4)
    print("PASSED\n")


if __name__ == "__main__":
    demo_simple_add()
    demo_three_ops()
    demo_all_ops()
