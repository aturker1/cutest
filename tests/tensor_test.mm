#include "cutest/metal/backend/tensor.h"

#include <cassert>
#include <cstdio>

using cutest::metal::Tensor;

static void check_fill(const Tensor& tensor, float expected) {
    assert((tensor.shape() == Tensor::Shape{2, 3}));
    assert(tensor.buffer().length == 6 * sizeof(float));
    const auto* values = static_cast<const float*>(tensor.buffer().contents);
    for (int i = 0; i < 6; ++i) assert(values[i] == expected);
}

int main() {
    @autoreleasepool {
        check_fill(Tensor::zeros({2, 3}), 0.0f);
        check_fill(Tensor::ones({2, 3}), 1.0f);
        std::puts("PASS: Metal tensor zeros and ones");
    }
}
