#include "cutest/metal/backend/tensor.h"

#include <cassert>
#include <cstdint>
#include <cstdio>

using cutest::metal::DType;
using cutest::metal::Tensor;

template <typename T>
static void check_fill(const Tensor& tensor, DType dtype, T expected) {
    assert((tensor.shape() == Tensor::Shape{2, 3}));
    assert(tensor.dtype() == dtype);
    assert(tensor.buffer().length == 6 * sizeof(T));
    const auto* values = static_cast<const T*>(tensor.buffer().contents);
    for (int i = 0; i < 6; ++i) assert(values[i] == expected);
}

int main() {
    @autoreleasepool {
        check_fill(Tensor::zeros({2, 3}), DType::FP32, 0.0f);
        check_fill(Tensor::ones({2, 3}, DType::FP32), DType::FP32, 1.0f);
        check_fill(Tensor::zeros({2, 3}, DType::BF16), DType::BF16, std::uint16_t{0x0000});
        check_fill(Tensor::ones({2, 3}, DType::BF16), DType::BF16, std::uint16_t{0x3f80});
        std::puts("PASS: FP32 and BF16 Metal tensor zeros and ones");
    }
}
