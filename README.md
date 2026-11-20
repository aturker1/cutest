# cutest

Minimal Metal tensor backend. Include `cutest/metal/backend/tensor.h` from
Objective-C++ (C++17, ARC, `-framework Metal`).

```cpp
using cutest::metal::DType;
using cutest::metal::Tensor;
auto a = Tensor::ones({2, 3}, DType::BF16);
auto b = Tensor::zeros({2, 3});  // FP32 by default.
```

Tensors store row-major values in shared Metal buffers: 4 bytes per FP32
element or 2 per BF16 element. BF16 fills use Apple Clang's `__bf16` type.
Use `shape()`, `dtype()`, and `buffer()` to inspect a tensor.

## Test

On macOS 11+ with Xcode Command Line Tools, run from the repo root
(tested with Bazel 9.3.0):

```sh
bazel test //:tensor_test
```
