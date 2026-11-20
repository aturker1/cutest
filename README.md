# cutest

Minimal Metal tensor backend. Include `cutest/metal/backend/tensor.h` from
Objective-C++ (C++17, ARC, `-framework Metal`).

```cpp
using cutest::metal::Tensor;
auto a = Tensor::ones({2, 3});
auto b = Tensor::zeros({2, 3});
```

Tensors store row-major FP32 values in shared Metal buffers. `shape()` and
`buffer()` expose the shape and storage.

## Test

On macOS 11+ with Xcode Command Line Tools, run from the repo root
(tested with Bazel 9.3.0):

```sh
bazel test //:tensor_test
```
