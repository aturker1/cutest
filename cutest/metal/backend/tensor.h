#pragma once

#import <Metal/Metal.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cutest::metal {

enum class DType { FP32, BF16 };

class Tensor {
public:
    using Shape = std::vector<std::size_t>;

    static Tensor zeros(Shape shape, DType dtype = DType::FP32) {
        return Tensor(std::move(shape), dtype, 0.0f);
    }
    static Tensor ones(Shape shape, DType dtype = DType::FP32) {
        return Tensor(std::move(shape), dtype, 1.0f);
    }

    const Shape& shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    id<MTLBuffer> buffer() const { return buffer_; }

private:
    Tensor(Shape shape, DType dtype, float value) : shape_(std::move(shape)), dtype_(dtype) {
        // Zero-length tensors need no allocation; an empty shape is a scalar.
        if (std::find(shape_.begin(), shape_.end(), 0) != shape_.end()) return;

        static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("Metal device unavailable");

        const bool bf16 = dtype_ == DType::BF16;
        const auto element_size = bf16 ? sizeof(__bf16) : sizeof(float);
        const auto max_elements = device.maxBufferLength / element_size;

        std::size_t elements = 1;
        for (auto dim : shape_) {
            if (elements > max_elements / dim) throw std::length_error("Metal tensor too large");
            elements *= dim;
        }
        buffer_ = [device newBufferWithLength:elements * element_size
                                       options:MTLResourceStorageModeShared];
        if (!buffer_) throw std::runtime_error("Metal tensor allocation failed");
        
        if (bf16) {
            std::fill_n(static_cast<__bf16*>(buffer_.contents), elements, static_cast<__bf16>(value));
        } else {
            std::fill_n(static_cast<float*>(buffer_.contents), elements, value);
        }
    }

    Shape shape_;
    DType dtype_;
    id<MTLBuffer> buffer_ = nil;
};

} // namespace cutest::metal
